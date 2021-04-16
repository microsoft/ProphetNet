# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from fairseq import utils
import math

def ngram_attention_bias(length, num_skip):
        bias_result = []
        for n_skip in range(num_skip):
            bias_n_skip = []
            for i in range(length):
                bias_this = [float('-inf')] * (2 * length)
                bias_this[length+i] = 0
                first_k = i - n_skip
                first_k = first_k if first_k > 0 else 0
                for j in range(first_k+1):
                    bias_this[j] = 0
                bias_n_skip.append(bias_this)
            bias_result.append(bias_n_skip)
        return torch.from_numpy(np.array(bias_result, dtype=np.float32))


class NgramMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False,ngram=2, num_buckets=32, relative_max_distance=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.ngram = ngram

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        self.relative_linear = nn.Linear(embed_dim, num_buckets * num_heads)
        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def _relative_positions_bucket(self, relative_positions, bidirectional=False):
        num_buckets = self.num_buckets
        max_distance = self.relative_max_distance
        n = -relative_positions
        result = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            result = result + torch.lt(n, torch.zeros_like(n)).int() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = torch.lt(n, max_exact)
        val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (
                    num_buckets - max_exact)
        val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1))
        val_if_large = val_if_large.int()
        result = result + torch.where(is_small, n.int(), val_if_large)
        return result


    def main_stream_relative_logits(self,query, attn_weights, real_positions,i_bucket_main_stream):
        # input query [T,B,C]
        # input attn_weights [T*head,T,S]
        # input real_positions [B,T] or [1,1]

        T,B,_ = query.size()
        S = attn_weights.size(-1)

        if i_bucket_main_stream is not None:
            i_buckets = i_bucket_main_stream
        else:
            # [B,T,S]
            relative_positions = torch.arange(1, S+1).unsqueeze(0).unsqueeze(0).repeat(B,T,1).to(real_positions.device)
            # [B,T,1]
            real_positions = real_positions.unsqueeze(0).repeat(B,T,1)
            # [B,T,S]
            relative_positions = relative_positions - real_positions
            # [B,T,T]
            i_buckets = self._relative_positions_bucket(relative_positions, False)

        # [B,T,C]
        query = query.transpose(0,1)
        # [B,T,Buckets*head]
        values = self.relative_linear(query)
        # [B,T,Buckets,head]
        values = values.view(values.size(0),values.size(1),self.num_buckets, self.num_heads)
        # [B,head,Buckets,T]
        values = values.transpose(1,3)
        # [B,head,T,Buckets]
        values = values.transpose(2,3)
        # [B*head,T,Buckets]
        values = values.reshape(attn_weights.size(0),attn_weights.size(1),-1)

        # => [B,head*T,T] => [B*head,T,T]
        i_buckets = i_buckets.repeat(1,self.num_heads,1).view(attn_weights.size(0),attn_weights.size(1),-1)
        # [B*head*T,Buckets]
        values = values.reshape(-1, values.size(-1))
        # [B*head*T,T]
        i_buckets = i_buckets.view(-1, i_buckets.size(-1)).long()
        # [B*head*T,T]
        result = torch.gather(values,dim=1,index=i_buckets)
        # [B*head,T,T]
        result = result.view(attn_weights.size(0),attn_weights.size(1),-1)

        return result

    def ngram_relative_logits(self, query, attn_weights, real_positions, i_bucket_relative_stream):
        # input query [ngram, T,B,C]
        # input attn_weights [ngram, B*head,T,S]
        # input real_positions [B,T] or [1,1]
        # input i_bucket_relative_stream [B,T, 2*T] or None

        N, T, B, _ = query.size()
        _, BH, _, S = attn_weights.size()

        if i_bucket_relative_stream is not None:
            i_buckets = i_bucket_relative_stream
        else:
            # [B,T,S]
            assert real_positions[0][0] == S - 1, 'memory position is 1 2 3 4 5(S-1)'
            relative_positions = torch.arange(0, S).unsqueeze(0).unsqueeze(0).repeat(B,T,1).to(real_positions.device)
            # print('relative_positions', relative_positions)
            # [B,T,1]
            real_positions = real_positions.unsqueeze(0).repeat(B,T,1)
            relative_positions = relative_positions
            # [B,T,2*T] or [B,T,S]
            relative_positions = relative_positions - real_positions
            i_buckets = self._relative_positions_bucket(relative_positions, False)

        # [ngram, B, T, C]
        query = query.transpose(1,2)
        # [ngram, B, T, bucket*head]
        values = self.relative_linear(query)
        # [ngram, B, T, bucket, head]
        values = values.view(*values.size()[:-1],self.num_buckets, self.num_heads)
        # [ngram, B, head, T, bucket]
        values = values.permute(0, 1, 4, 2, 3)
        # [ngram*B*head, T, bucket]
        values = values.reshape(N*BH,T,-1)

        # [ngram, B, head*T, S]
        i_buckets = i_buckets.unsqueeze(0).repeat(N,1,self.num_heads,1)

        values = values.reshape(-1, values.size(-1))
        i_buckets = i_buckets.view(-1, i_buckets.size(-1)).long()
        # [ngram*B*head*T, S]
        result = torch.gather(values,dim=1,index=i_buckets)
        # [ngram, B*head, T, S]
        result = result.view(N, BH , T, -1)

        return result

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False,
                self_attn_mask=None,
                ngram_mask_matrix=None,
                i_buckets_main_stream=None,
                i_bucket_relative_stream=None,
                real_positions=None
                ):
        """Input shape: Time x Batch x Channel

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        q, k, v = self.in_proj_qkv(query)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # h: 输入的隐状态
        h_list = query.chunk(1 + self.ngram, dim=0)

        q_list = q.chunk(1 + self.ngram, dim=1)
        k_list = k.chunk(1 + self.ngram, dim=1)
        v_list = v.chunk(1 + self.ngram, dim=1)

        h_main, h_predict_list = h_list[0], h_list[1:]
        q_main, q_predict_list = q_list[0], q_list[1:]
        k_main, k_predict_list = k_list[0], k_list[1:]
        v_main, v_predict_list = v_list[0], v_list[1:]

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    assert False, 'static_kv not supprt in ngram decoder'
                    k = prev_key
                else:
                    k_main = torch.cat((prev_key, k_main), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v_main = torch.cat((prev_value, v_main), dim=1)
            saved_state['prev_key'] = k_main.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v_main.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        real_tgt_len = tgt_len // (1 + self.ngram)

        attn_weights_main = torch.bmm(q_main, k_main.transpose(1, 2))

        main_relative_logits = self.main_stream_relative_logits(h_main, attn_weights_main,real_positions, i_buckets_main_stream)
        attn_weights_main = attn_weights_main + main_relative_logits

        if self_attn_mask is not None:
            self_attn_mask = self_attn_mask.unsqueeze(0)
            attn_weights_main = attn_weights_main + self_attn_mask

        attn_weights_main = utils.softmax(
            attn_weights_main, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights_main)
        attn_weights_main = F.dropout(attn_weights_main, p=self.dropout, training=self.training)

        attn_main = torch.bmm(attn_weights_main, v_main)
        attn_main = attn_main.transpose(0, 1).contiguous().view(1, real_tgt_len, bsz, embed_dim)
        attn_main = self.out_proj(attn_main)


        # [ngram, B*head, T, c]
        q_ngram = torch.cat(q_predict_list, 0).view(self.ngram, -1, real_tgt_len, self.head_dim)
        # [ngram, B*head, 2*T, c]
        k_ngram = torch.cat([torch.cat([k_main, k_p], 1).unsqueeze(0) for k_p in k_predict_list], 0)
        # below code slower than above for loop
        # k_ngram = torch.cat([k_main.unsqueeze(0).repeat(self.ngram, 1, 1, 1) , torch.cat(k_predict_list).view(self.ngram, -1, real_tgt_len, self.head_dim)], 2)

        # [ngram, T, B, C]
        h_ngram = torch.cat(h_predict_list, 0).view(self.ngram, real_tgt_len, bsz, embed_dim)

        # [ngram, B*head, 2*T, c]
        v_ngram = torch.cat([torch.cat([v_main, v_p], 1).unsqueeze(0) for v_p in v_predict_list], 0)
        # below code slower than above for loop
        # v_ngram = torch.cat([v_main.unsqueeze(0).repeat(self.ngram, 1, 1, 1) , torch.cat(v_predict_list).view(self.ngram, -1, real_tgt_len, self.head_dim)], 2)

        # [ngram, B*head, T, 2*T]
        attn_weights_ngram = torch.einsum('nbtc,nbsc->nbts', (q_ngram, k_ngram))

        # [ngram, B*head, T, S]
        predict_relative_logits = self.ngram_relative_logits(h_ngram, attn_weights_ngram, real_positions, i_bucket_relative_stream)
        # [ngram, B*head, T, 2*T]
        attn_weights_ngram = attn_weights_ngram + predict_relative_logits

        if ngram_mask_matrix is not None:
            ngram_mask_matrix = ngram_mask_matrix.unsqueeze(1)
            attn_weights_ngram = attn_weights_ngram + ngram_mask_matrix

        attn_weights_ngram = utils.softmax(
            attn_weights_ngram, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights_ngram)
        attn_weights_ngram = F.dropout(attn_weights_ngram, p=self.dropout, training=self.training)

        # [ngram, B*head, T, c]
        attn_ngram = torch.einsum('nbts,nbsc->nbtc', (attn_weights_ngram, v_ngram))
        # [ngram, T, B, C]
        attn_ngram = attn_ngram.transpose(1, 2).contiguous().view(self.ngram, real_tgt_len, bsz, embed_dim)
        attn_ngram = self.out_proj(attn_ngram)

        attn_result = []
        attn_result.append(attn_main)
        attn_result.append(attn_ngram)

        # [1+ngram*T, B, C]
        attn = torch.cat(attn_result, 0).view(-1, bsz, embed_dim)
        return attn, None

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )
