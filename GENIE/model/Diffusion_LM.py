import torch
from torch import nn
import numpy as np
import math
from transformers import (
    BertModel,
    BertConfig,
)
from model.CrossAttentionTransformers import BasicTransformerBlock

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Diffusion_LM(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            dropout=0,
            config=None,
            config_name='bert-base-uncased',
            vocab_size=None,
            init_pretrained=True,
            logits_mode=1,
            token_emb_type='pretrain',
            # num_heads=1,
            # channel_mult=(1, 2, 4, 8),
            # use_scale_shift_norm=False,
            # training_mode='emb',
            # experiment_mode='lm',
            # num_heads_upsample=-1,
            # use_checkpoint=False,
            # num_classes=None,
            # conv_resample=True,
            # attention_resolutions,
            # num_res_blocks,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.init_pretrained = init_pretrained
        self.token_emb_type = token_emb_type

        config = BertConfig.from_pretrained(config_name)
        config.hidden_dropout_prob = self.dropout
        print(config)

        # 可训练的 embedding 层
        self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
        # position embedding
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if self.token_emb_type == 'pretrain':
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            self.word_embedding.weight = temp_bert.embeddings.word_embeddings.weight
            self.position_embeddings.weight = temp_bert.embeddings.position_embeddings.weight
        elif self.token_emb_type == 'random':
            print("load embedding weight random")
        else:
            return NotImplementedError

        if self.logits_mode == 2:
            # self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=False)
            self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=True)
        else:
            self.lm_head = nn.Linear(self.in_channels, vocab_size)

        # share weight between lm_head and word_embedding
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        # self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
        # self.lm_head = nn.Linear(self.in_channels, vocab_size)
        # with th.no_grad():
        #     self.lm_head.weight = self.word_embedding.weight

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size),
        )

        # position embedding
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # # label embedding
        # if self.num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # input transform
        self.input_up_proj = nn.Sequential(
            nn.Linear(in_channels, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        '''
        Diffusion Transformer (6 layer)
        '''
        if self.init_pretrained:
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            del temp_bert.embeddings
            del temp_bert.pooler
            self.input_transformers = temp_bert.encoder
            print('initializing from pretrained bert.')
        else:
            temp_bert = BertModel(config)
            self.input_transformers = temp_bert.encoder
            print('initializing from random bert.')

        # output transform
        self.output_down_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, out_channels)
        )

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()

            #
            # scores1 = th.cdist(self.lm_head.weight.unsqueeze(0), hidden_repr, p=2)
            # scores1 = -scores1.permute(0, 2, 1).contiguous()
            #
            # print(scores1.shape, scores.shape)
            # print(scores1[0,0], scores[0,0])
            # print(torch.isclose(scores1, scores))

            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps, attention_mask=None, y=None, src_ids=None, src_mask=None):

        # prepare embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]
        # print(emb_x.shape, emb.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        # encode embedding
        # print(emb_inputs.shape, attention_mask.shape)
        input_trans_hidden_states = self.input_transformers(emb_inputs, attention_mask=attention_mask).last_hidden_state
        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        return h



class CrossAttention_Diffusion_LM(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            dropout=0,
            config=None,
            config_name='bert-base-uncased',
            vocab_size=None,
            init_pretrained=True,
            logits_mode=1,
            token_emb_type='pretrain',
            fix_encoder=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.init_pretrained = init_pretrained
        self.token_emb_type = token_emb_type
        self.fix_encoder = fix_encoder

        cfg = BertConfig.from_pretrained(config_name)
        cfg.num_hidden_layers = 6
        self.passage_encoder = BertModel.from_pretrained(config_name, config=cfg)
        # self.passage_encoder = BertModel.from_pretrained(
        #     "/colab_space/Lin0/PROD/KDexp/pretrain_model/bert-base-uncased", config=cfg)




        config = BertConfig.from_pretrained(config_name)
        config.hidden_dropout_prob = self.dropout
        print(config)

        # trainable embedding layer
        self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
        # position embedding
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if self.logits_mode == 2:
            # self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=False)
            self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=True)
        else:
            self.lm_head = nn.Linear(self.in_channels, vocab_size)

        # share weight between lm_head and word_embedding
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        # self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
        # self.lm_head = nn.Linear(self.in_channels, vocab_size)
        # with th.no_grad():
        #     self.lm_head.weight = self.word_embedding.weight

        # time embedding layer
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size),
        )

        # position embedding
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # # label embedding
        # if self.num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # input transform
        self.input_up_proj = nn.Sequential(
            nn.Linear(in_channels, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        config.num_hidden_layers = 6
        # define cross attention transformer block(6 layer)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.hidden_size // config.num_attention_heads,
                    dropout=config.hidden_dropout_prob,
                    cross_attention_dim=config.hidden_size,
                    activation_fn="geglu",
                )
                for d in range(config.num_hidden_layers)
            ]
        )

        # output transform
        self.output_down_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, out_channels)
        )

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()

            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps, src_input_ids, src_attention_mask, attention_mask=None,
                answer_id=None, answer_mask=None, y=None, src_ids=None, src_mask=None):

        # prepare embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]
        # print(emb_x.shape, emb.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        hidden_states = self.dropout(self.LayerNorm(emb_inputs))
        # encode embedding
        # print(emb_inputs.shape, attention_mask.shape)
        if self.fix_encoder:
            with torch.no_grad():
                out = self.passage_encoder(input_ids=src_input_ids,
                                                 attention_mask=src_attention_mask)
                passage_hidden = out.last_hidden_state
        else:
            out = self.passage_encoder(input_ids=src_input_ids,
                                       attention_mask=src_attention_mask)
            passage_hidden = out.last_hidden_state + 0 * out.pooler_output.unsqueeze(1)

        if answer_id is not None:
            answer_hidden_states = hidden_states.clone()
            answer_out = self.passage_encoder(input_ids=answer_id,
                                              attention_mask=answer_mask)
            answer_hidden = answer_out.last_hidden_state + 0 * answer_out.pooler_output.unsqueeze(1)
            for block in self.transformer_blocks:
                answer_hidden_states = block(answer_hidden_states, answer_hidden)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, passage_hidden)

        if answer_id is not None:
            # print("model_qg_forward...")
            hidden_states = hidden_states + answer_hidden_states

        h = self.output_down_proj(hidden_states)
        h = h.type(x.dtype)
        return h






