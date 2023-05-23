
from hmac import new
import token
from turtle import pos
from transformers import RobertaModel, RobertaPreTrainedModel, LongformerModel, RobertaConfig, LongformerConfig, LongformerPreTrainedModel, BartPretrainedModel, BartModel
from transformers.models.bart.modeling_bart import BartEncoder
from transformers import PreTrainedModel
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import MarginRankingLoss, BCEWithLogitsLoss
from torch.nn.parameter import Parameter
from packaging import version
import numpy as np
import torch.nn.functional as F

class RobertaRankerHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        """
        features: (B, C, L, D)  B for batch size, C for candidate number, L for sequencen length, D for hiddensize
        """
        x = features[:, :, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x) # (B, C, 1)
        return x


class BartRankerHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        """
        features: (B, C, L, D)  B for batch size, C for candidate number, L for sequencen length, D for hiddensize
        """
        x = features[:, :, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x) # (B, C, 1)
        return x



class RobertaRanker(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]


    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.classifier = RobertaRankerHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        input_ids: (B, C, L) B for batch size, C for number of candidates, L for sequence length
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, candidate_num, seq_len = input_ids.size()
        
        input_ids = input_ids.view(-1, input_ids.size(-1)) #(B * C, L)
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        # this part of code is added due to roberta has a buffered token_type_ids that may not suitable for the new input length
        # we create a new token_type_ids to the model, though they are still 0s
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] # (B*C, L, D)

        sequence_output = sequence_output.view(batch_size, candidate_num, seq_len, -1) # (B, C, L, D)

        logits = self.classifier(sequence_output).squeeze(-1) # (B, C)

        loss = None
        if self.loss_type == "rank":
            ones = torch.ones_like(logits)
            loss_func = MarginRankingLoss(0.0)
            loss = loss_func(logits, logits, ones)
            for i in range(1, candidate_num):
                # i is the gap
                pos_score = logits[:, :-i] # (B, C - i) ranked higher
                neg_score = logits[:, i:] # (B, C- i ) ranked lower
                pos_score = pos_score.contiguous().view(-1)
                neg_score = neg_score.contiguous().view(-1) 

                ones = torch.ones_like(pos_score)
                loss_func = MarginRankingLoss(0.01 * i)
                l = loss_func(pos_score, neg_score, ones)
                loss += l
        elif self.loss_type == 'binary':
            labels = torch.zeros_like(logits, dtype=torch.float32) # (B, C)
            labels[:,0] = 1
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits, labels)
        elif self.loss_type == 'contrastive':
            logit_soft = F.softmax(logits, dim = -1)
            pos_probs = logit_soft[:, 0]
            loss = -torch.log(pos_probs).mean()


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _resize_position_embedding(self, new_max_position_embedding, extend_way = 'normal'):
        '''
            resize the model's position embedding to match the new positional embedding
            should also update the config.max_position_ebmddings here for avoiding error when loading finetuned checkpoints
            should also change the token_type_ids registed in embedding layers
        '''
        # self.roberta.embeddings.position_embeddings
        old_weights = self.roberta.embeddings.position_embeddings.weight # tensor
        old_num_emb, embedding_dim = old_weights.size()

        if extend_way == 'normal':
            # initialize new weight in normal
            added_weights = torch.empty((new_max_position_embedding - old_num_emb, embedding_dim), requires_grad=True)
            nn.init.normal_(added_weights)
            new_weights = torch.cat((old_weights, added_weights), 0)
        elif extend_way == 'copy':
            # initialize new weight by copying from the old weights
            # to be implemented
            len_to_extend = new_max_position_embedding - old_num_emb
            old_weight_np = old_weights.detach().numpy()

            added_weights = np.array(old_weight_np[2: len_to_extend % (old_num_emb - 2) +2])
            for _ in range(len_to_extend // (old_num_emb - 2) ):
                added_weights = np.concatenate(added_weights, np.array(old_weight_np[2:]))
            
            added_weights = torch.Tensor(added_weights)
            added_weights.requires_grad = True
            new_weights = torch.cat((old_weights, added_weights), 0)

        self.roberta.embeddings.position_embeddings = nn.Embedding(new_max_position_embedding, embedding_dim,
                                                    padding_idx = self.roberta.embeddings.position_embeddings.padding_idx,
                                                    _weight = new_weights)

        self.config.max_position_embeddings = new_max_position_embedding





class LongformerRanker(LongformerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]


    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = RobertaRankerHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        input_ids: (B, C, L) B for batch size, C for number of candidates, L for sequence length
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, candidate_num, seq_len = input_ids.size()
        
        input_ids = input_ids.view(-1, input_ids.size(-1)) #(B * C, L)
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:,0] = 1

        # this part of code is added due to roberta has a buffered token_type_ids that may not suitable for the new input length
        # we create a new token_type_ids to the model, though they are still 0s
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask = global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] # (B*C, L, D)

        sequence_output = sequence_output.view(batch_size, candidate_num, seq_len, -1) # (B, C, L, D)

        logits = self.classifier(sequence_output).squeeze(-1) # (B, C)

        loss = None
        if self.loss_type == "rank":
            ones = torch.ones_like(logits)
            loss_func = MarginRankingLoss(0.0)
            loss = loss_func(logits, logits, ones)
            for i in range(1, candidate_num):
                # i is the gap
                pos_score = logits[:, :-i] # (B, C - i) ranked higher
                neg_score = logits[:, i:] # (B, C- i ) ranked lower
                pos_score = pos_score.contiguous().view(-1)
                neg_score = neg_score.contiguous().view(-1) 

                ones = torch.ones_like(pos_score)
                loss_func = MarginRankingLoss(0.01 * i)
                l = loss_func(pos_score, neg_score, ones)
                loss += l
        elif self.loss_type == 'binary':
            labels = torch.zeros_like(logits, dtype=torch.float32) # (B, C)
            labels[:,0] = 1
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits, labels)
        elif self.loss_type == 'contrastive':
            logit_soft = F.softmax(logits, dim = -1)
            pos_probs = logit_soft[:, 0]
            loss = -torch.log(pos_probs).mean()



        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # def _resize_position_embedding(self, new_max_position_embedding, extend_way = 'normal'):
    #     '''
    #         resize the model's position embedding to match the new positional embedding
    #         should also update the config.max_position_ebmddings here for avoiding error when loading finetuned checkpoints
    #         should also change the token_type_ids registed in embedding layers
    #     '''
    #     # self.roberta.embeddings.position_embeddings
    #     old_weights = self.roberta.embeddings.position_embeddings.weight # tensor
    #     old_num_emb, embedding_dim = old_weights.size()

    #     if extend_way == 'normal':
    #         # initialize new weight in normal
    #         added_weights = torch.empty((new_max_position_embedding - old_num_emb, embedding_dim), requires_grad=True)
    #         nn.init.normal_(added_weights)
    #         new_weights = torch.cat((old_weights, added_weights), 0)
    #     elif extend_way == 'copy':
    #         # initialize new weight by copying from the old weights
    #         # to be implemented
    #         len_to_extend = new_max_position_embedding - old_num_emb
    #         old_weight_np = old_weights.detach().numpy()

    #         added_weights = np.array(old_weight_np[2: len_to_extend % (old_num_emb - 2) +2])
    #         for _ in range(len_to_extend // (old_num_emb - 2) ):
    #             added_weights = np.concatenate(added_weights, np.array(old_weight_np[2:]))
            
    #         added_weights = torch.Tensor(added_weights)
    #         added_weights.requires_grad = True
    #         new_weights = torch.cat((old_weights, added_weights), 0)

    #     self.roberta.embeddings.position_embeddings = nn.Embedding(new_max_position_embedding, embedding_dim,
    #                                                 padding_idx = self.roberta.embeddings.position_embeddings.padding_idx,
    #                                                 _weight = new_weights)

    #     self.config.max_position_embeddings = new_max_position_embedding



class BartRanker(BartModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]


    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)

        self.classifier = BartRankerHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        input_ids: (B, C, L) B for batch size, C for number of candidates, L for sequence length
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, candidate_num, seq_len = input_ids.size()
        
        input_ids = input_ids.view(-1, input_ids.size(-1)) #(B * C, L)
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        # this part of code is added due to roberta has a buffered token_type_ids that may not suitable for the new input length
        # we create a new token_type_ids to the model, though they are still 0s
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] # (B*C, L, D)

        sequence_output = sequence_output.view(batch_size, candidate_num, seq_len, -1) # (B, C, L, D)

        logits = self.classifier(sequence_output).squeeze(-1) # (B, C)

        loss = None
        if self.loss_type == "rank":
            ones = torch.ones_like(logits)
            loss_func = MarginRankingLoss(0.0)
            loss = loss_func(logits, logits, ones)
            for i in range(1, candidate_num):
                # i is the gap
                pos_score = logits[:, :-i] # (B, C - i) ranked higher
                neg_score = logits[:, i:] # (B, C- i ) ranked lower
                pos_score = pos_score.contiguous().view(-1)
                neg_score = neg_score.contiguous().view(-1) 

                ones = torch.ones_like(pos_score)
                loss_func = MarginRankingLoss(0.01 * i)
                l = loss_func(pos_score, neg_score, ones)
                loss += l
        elif self.loss_type == 'binary':
            labels = torch.zeros_like(logits, dtype=torch.float32) # (B, C)
            labels[:,0] = 1
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits, labels)
        elif self.loss_type == 'contrastive':
            logit_soft = F.softmax(logits, dim = -1)
            pos_probs = logit_soft[:, 0]
            loss = -torch.log(pos_probs).mean()


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    