# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion 

@register_criterion('ngram_language_loss')
class NgramLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        logits_list = model(**sample['net_input'], return_all_hiddens=False)[0]
        targets = model.get_targets(sample, [logits_list[0]])


        ngram = len(logits_list)
        # [B, ngram, T]
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break

            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i,:,:] = targets
        targets = expend_targets

        logits = torch.cat(logits_list, dim=0) #.view(ngram, *logits_list[0].size())

        lprobs = F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )

        loss = F.nll_loss(
               lprobs,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )

        if self.eps > 0.:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * loss + eps_i * smooth_loss

        sample_size = targets.ne(self.padding_idx).int().sum().item()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
