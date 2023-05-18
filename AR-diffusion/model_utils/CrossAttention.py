import torch

import torch.nn.functional as F

from torch import nn
from typing import Optional


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        config,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        attention_dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        
        self.r_w_bias, self.r_r_bias = None, None
        if self.config.time_att:
            # self.layernorm = nn.LayerNorm(config.time_channels, eps=1e-12)
            self.dropout = nn.Dropout(dropout)
            self.time_trans = nn.Sequential(
                nn.Linear(config.time_channels, config.time_channels * 4),
                nn.SiLU(),
                nn.Linear(config.time_channels * 4, num_attention_heads * attention_head_dim),
            )

            # self.time_net = nn.Linear(self.dim, num_attention_heads * attention_head_dim, bias=attention_bias)
            
            if config.att_strategy == 'txl':
                self.r_w_bias = nn.Parameter(torch.rand(num_attention_heads, attention_head_dim))
                self.r_r_bias = nn.Parameter(torch.rand(num_attention_heads, attention_head_dim))

        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            bias=attention_bias,
            att_strategy=config.att_strategy,
        )  # self-attention
        
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            bias=attention_bias,
            att_strategy=config.att_strategy,
        )

        # layer norms
        self.norm1 = nn.LayerNorm(dim, eps=1e-12)
        self.norm2 = nn.LayerNorm(dim, eps=1e-12)
        self.norm3 = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, hidden_states, context,
                time_emb=None, encoder_key_padding_mask=None, tgt_padding_mask=None):
        # time_emb: no transform [bs, seq_len, hz]
        if time_emb is not None:
            # time_emb = (self.time_trans(self.dropout(self.layernorm(time_emb))))
            time_emb = (self.time_trans(self.dropout(time_emb)))  # [bs, seq_len, n_head*head_d]
            # if self.config.att_strategy == 'txl':
            #     time_emb = self.time_net(time_emb)

            batch_size, seq_len, _ = time_emb.shape
            time_emb = time_emb.reshape(
                batch_size, seq_len, self.num_attention_heads, self.dim // self.num_attention_heads
            )
            if self.config.att_strategy == 'rotary':
                time_emb = time_emb.chunk(2, dim=-1)
            elif self.config.att_strategy == 'txl':
                time_emb = time_emb.permute(0, 2, 1, 3).reshape(
                    batch_size * self.num_attention_heads, seq_len, self.dim // self.num_attention_heads)

        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn1(
            norm_hidden_states,
            time_embeddings=time_emb,
            tgt_padding_mask=tgt_padding_mask,
            r_w_bias=self.r_w_bias,
            r_r_bias=self.r_r_bias,
        ) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(
            norm_hidden_states,
            context=context,
            key_padding_mask=encoder_key_padding_mask,
        ) + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(norm_hidden_states) + hidden_states

        return hidden_states


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bias=False,
        att_strategy=None,
    ):
        super().__init__()
        self.att_strategy = att_strategy
        self.inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        
        self.dropatt = nn.Dropout(attention_dropout)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor, time_embeddings=None):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len,
                                head_size, dim // head_size)

        if time_embeddings is not None:
            tensor = self.apply_rotary(tensor, time_embeddings)

        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size,
                                head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_len, dim * head_size)
        return tensor

    @staticmethod
    def apply_rotary(x, embeddings):
        cos, sin = embeddings
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)

    def forward(self, hidden_states, time_embeddings=None, context=None, 
                key_padding_mask=None, tgt_padding_mask=None, r_w_bias=None, r_r_bias=None):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        if context is None:
            context = hidden_states

        query = self.to_q(hidden_states)
        if self.att_strategy == 'rotary' and time_embeddings is not None:
            # [bs*n_head, q_len, head_d]
            query = self.reshape_heads_to_batch_dim(query, time_embeddings=time_embeddings)
        elif self.att_strategy == 'txl' and time_embeddings is not None:
            # [bs, q_len, n_head*head_d] -> [bs, q_len, n_head, head_d]
            query = query.reshape(batch_size, sequence_length, self.heads, hidden_size // self.heads)
        else:
            query = self.reshape_heads_to_batch_dim(query)
        
        # self attention
        key = self.to_k(context)
        value = self.to_v(context)
        
        if self.att_strategy == 'rotary' and time_embeddings is not None:
            # [bs*n_head, k_len, head_d]
            key = self.reshape_heads_to_batch_dim(key, time_embeddings=time_embeddings)
            value = self.reshape_heads_to_batch_dim(value, time_embeddings=time_embeddings)
        else:
            # [bs*n_head, k_len, head_d]
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

        if key_padding_mask is not None:
            key_padding_mask = torch.repeat_interleave(key_padding_mask, self.heads, 0)
        elif tgt_padding_mask is not None:
            tgt_padding_mask = torch.repeat_interleave(tgt_padding_mask, self.heads, 0)

        hidden_states = self._attention(query, key, value,
                                        key_padding_mask=key_padding_mask, 
                                        tgt_padding_mask=tgt_padding_mask,
                                        time_embeddings=time_embeddings,
                                        r_w_bias=r_w_bias,
                                        r_r_bias=r_r_bias,)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def _attention(self, query, key, value, 
                   key_padding_mask=None, tgt_padding_mask=None, time_embeddings=None, 
                   r_w_bias=None, r_r_bias=None):
        # self_attention
        if self.att_strategy == 'txl' and (r_w_bias is not None and r_r_bias is not None):
            batch_size, seq_len, _, _, = query.shape
            
            r_w_query = query + r_w_bias
            # [bs, q_len, n_head, head_d] -> [bs*n_head, q_len, head_d]
            r_w_query = r_w_query.permute(0, 2, 1, 3).reshape(
                batch_size * self.heads, seq_len, self.dim_head)
            AC = torch.matmul(r_w_query, key.transpose(-1, -2))  # [bs*n_head, q_len, k_len]

            r_r_query = query + r_r_bias
            # [bs, q_len, n_head, head_d] -> [bs*n_head, q_len, head_d]
            r_r_query = r_r_query.permute(0, 2, 1, 3).reshape(
                batch_size * self.heads, seq_len, self.dim_head)
            BD = torch.matmul(r_r_query, time_embeddings.transpose(-1, -2))  # [bs*n_head, q_len, k_len]

            attention_scores = (AC + BD) * self.scale  # [bs*n_head, q_len, k_len]
        else:
            # [bs*n_head q_len, k_len]
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        
        if key_padding_mask is not None:
            attention_scores = attention_scores.masked_fill(
                key_padding_mask.unsqueeze(1).eq(0),
                -float("inf")
            )
            
        elif tgt_padding_mask is not None:
            attention_scores = attention_scores.masked_fill(
                tgt_padding_mask.unsqueeze(1).eq(0),
                -float("inf")
            )
        
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.dropatt(attention_probs)
        
        # compute attention output
        hidden_states = torch.matmul(attention_probs, value)  # [bs*n_head, q_len, head_d]

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "geglu":
            geglu = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            geglu = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(geglu)
        # activation dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    """
    The approximate form of Gaussian Error Linear Unit (GELU)

    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)
