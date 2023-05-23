from dataclasses import dataclass
from lib2to3.pgen2 import token
import torch
import numpy as np
from collections import Counter
from rouge_score import rouge_scorer, scoring
from dataclasses import dataclass
from datasets import  load_metric
from .coqa_evaluator import CoQAEvaluator
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import nltk
import random


class compute_rouge:
    def __init__(self, tokenizer = None, ignore_pad_token_for_loss = True):
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.metric = load_metric('rouge')
        self.scorer = rouge_scorer.RougeScorer(rouge_types = ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def __call__(self, eval_preds, decoded = False):
        preds, labels = eval_preds
        if not decoded:
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            if self.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(preds, labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result




class compute_qg:
    def __init__(self, tokenizer = None, ignore_pad_token_for_loss = True):
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.rouge_metric = load_metric('rouge')
        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types =  ["rougeL"], use_stemmer=True)
        self.bleu_scorer = load_metric('bleu')
        self.meteor_scorer = load_metric('meteor')

    def postprocess_text_bleu(self, preds, labels):
        preds = [nltk.word_tokenize(pred) for pred in preds]
        labels = [nltk.word_tokenize(label) for label in labels]
        
        return preds, labels

    def __call__(self, eval_preds, decoded = False):
        preds, labels = eval_preds
        if not decoded:
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            if self.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = {}
        # Some simple post-processing
        preds_bleu, labels_bleu = self.postprocess_text_bleu(preds, labels)
        # preds_meteor = [' '.join(pred) for pred in preds_bleu]
        # labels_meteor = [' '.join(label) for label in labels_bleu]

        result_rouge = self.rouge_metric.compute(predictions=preds, references=labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result['rougeL'] = result_rouge['rougeL'].mid.fmeasure * 100

        result['bleu_4'] = self.bleu_scorer._compute(preds_bleu, [[l] for l in labels_bleu], max_order=4)['bleu'] * 100

        result['meteor'] = self.meteor_scorer._compute(preds_bleu, labels_bleu)['meteor'] * 100

        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result





class compute_dialog:
    def __init__(self, tokenizer = None, ignore_pad_token_for_loss = True):
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.bleu_scorer = load_metric('bleu')

    def postprocess_text_bleu(self, preds, labels):
        preds = [nltk.word_tokenize(pred) for pred in preds]
        labels = [nltk.word_tokenize(label) for label in labels]

        return preds, labels

    def bleu(self, hyps, refs):
        """ Calculate bleu 1/2. """
        bleu_1 = []
        bleu_2 = []
        for hyp, ref in zip(hyps, refs):
            try:
                score = bleu_score.sentence_bleu(
                    [ref], hyp,
                    smoothing_function=SmoothingFunction().method7,
                    weights=[1, 0, 0, 0])
            except:
                score = 0
            bleu_1.append(score)
            try:
                score = bleu_score.sentence_bleu(
                    [ref], hyp,
                    smoothing_function=SmoothingFunction().method7,
                    weights=[0.5, 0.5, 0, 0])
            except:
                score = 0
            bleu_2.append(score)
        bleu_1 = np.average(bleu_1)
        bleu_2 = np.average(bleu_2)
        return bleu_1, bleu_2

    def distinct(self, seqs):
        """ Calculate intra/inter distinct 1/2. """
        batch_size = len(seqs)
        intra_dist1, intra_dist2 = [], []
        unigrams_all, bigrams_all = Counter(), Counter()
        for seq in seqs:
            unigrams = Counter(seq)
            bigrams = Counter(zip(seq, seq[1:]))
            intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
            intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

            unigrams_all.update(unigrams)
            bigrams_all.update(bigrams)

        inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
        inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
        intra_dist1 = np.average(intra_dist1)
        intra_dist2 = np.average(intra_dist2)
        return intra_dist1, intra_dist2, inter_dist1, inter_dist2

    def __call__(self, eval_preds, decoded = False):
        preds, labels = eval_preds
        if not decoded:
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            if self.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        preds_bleu, labels_bleu = self.postprocess_text_bleu(preds, labels)

        # Extract a few results from ROUGE
        result = {}
        bleu_1, bleu_2 = self.bleu(preds_bleu, labels_bleu)
        result['bleu_1'] =  bleu_1*100
        result['bleu_2']  = bleu_2*100
        _,_,d1,d2 = self.distinct(preds_bleu)
        result['distinct_1'] = d1*100
        result['distinct_2'] = d2*100

        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result





class compute_coqa:
    def __init__(self, tokenizer = None, ignore_pad_token_for_loss = True):
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss

    def postprocess_text_coqa(self, preds, labels):
        preds = [' '.join(nltk.word_tokenize(pred)) for pred in preds]
        labels = [' '.join(nltk.word_tokenize(label)) for label in labels]

        return preds, labels

    def distinct(self, seqs):
        """ Calculate intra/inter distinct 1/2. """
        batch_size = len(seqs)
        intra_dist1, intra_dist2 = [], []
        unigrams_all, bigrams_all = Counter(), Counter()
        for seq in seqs:
            unigrams = Counter(seq)
            bigrams = Counter(zip(seq, seq[1:]))
            intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
            intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

            unigrams_all.update(unigrams)
            bigrams_all.update(bigrams)

        inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
        inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
        intra_dist1 = np.average(intra_dist1)
        intra_dist2 = np.average(intra_dist2)
        return intra_dist1, intra_dist2, inter_dist1, inter_dist2

    def __call__(self, eval_preds, decoded = False):
        preds, labels = eval_preds
        if not decoded:
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            if self.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        preds_coqa, labels_coqa = self.postprocess_text_coqa(preds, labels)

        # Extract a few results from ROUGE
        result = {}
        result['f1'] = CoQAEvaluator.quick_model_performance(preds_coqa, labels_coqa) * 100

        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
