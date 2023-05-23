from dataclasses import dataclass
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


def clean(x):
    """
     this is for clean the output sequences
    """
    if x[0]=='.':
        x = x[1:]
    x = x.strip()
    return x

class compute_rouge:
    def __init__(self):
        self.metric = load_metric('rouge')
        self.scorer = rouge_scorer.RougeScorer(rouge_types = ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)

    def postprocess_text(self, preds, labels):
        preds = [clean(pred.strip()) for pred in preds]
        labels = [clean(label.strip()) for label in labels]

        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def __call__(self, eval_preds):
        preds, labels = eval_preds

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(preds, labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    def get_candidates(self, targets,  preds, num_cand, max_num, strategy):
        """
        args:
            targets: list of targets for each sample
            pos: list of positive samples for each sample
            preds: list of predictions, length == len(targets) * num_cand
            num_cand: number of candidates
            max_num: number of returned indices per sample 
        returns:
            indices: Torch tensor, (B * (C-1), ), since the positive candidate is not generated from the generator
            candiates: candidates, with the length of len(targets) * max_num
            NOTE: We should always keep the positive sequences in the first candidate for each sample
        """
        preds_processed, targets_processed = self.postprocess_text(preds, targets)

        indices = []
        candidates = []
        rewards = []
        for i,t in enumerate(targets_processed):
            scores = []
            ps = preds_processed[i * num_cand: (i+1)*num_cand]
            ps_nopro = preds[i * num_cand: (i+1)*num_cand] # for the candidates, we use no preprocessed version
            for j,p in enumerate(ps):
                s = self.scorer.score(t, p)
                scores.append((j + i * num_cand, s["rouge1"].fmeasure / 0.45 + s["rouge2"].fmeasure / 0.2 + s["rougeLsum"].fmeasure / 0.4, ps_nopro[j].strip()))

            scores = sorted(scores, key = lambda x: x[1], reverse=True)
            

            idx_this = [scores[0][0]] # the first as pos
            cand_this = [scores[0][2]]
            rewards_this = [scores[0][1]]
            scores = scores[1:]

            if strategy == 'random':
                s_for_pick = random.sample(scores, max_num - 1)
                idx_this +=  [s[0] for s in s_for_pick]
                cand_this +=  [s[2] for s in s_for_pick]
                rewards_this += [s[1] for s in s_for_pick]
            else:
                if strategy == 'top':
                    idx_this +=  [s[0] for s in scores[:max_num-1]]
                    cand_this +=  [s[2] for s in scores[:max_num-1]]
                    rewards_this += [s[1] for s in scores[:max_num-1]]
                elif strategy == 'bottom':
                    idx_this +=  [s[0] for s in scores[-max_num+1:]]
                    cand_this +=  [s[2] for s in scores[-max_num+1:]]
                    rewards_this += [s[1] for s in scores[-max_num+1:]]
                elif strategy == 'top-bottom':
                    n_top = (max_num-1) // 2
                    n_bottom = (max_num-1) - n_top
                    idx_this +=  [s[0] for s in scores[:n_top]]
                    cand_this += [s[2] for s in scores[:n_top]]
                    idx_this +=  [s[0] for s in scores[-n_bottom:]]
                    cand_this += [s[2] for s in scores[-n_bottom:]]
                    rewards_this += [s[1] for s in scores[:n_top]]
                    rewards_this += [s[1] for s in scores[-n_bottom:]]


            indices += idx_this
            candidates += cand_this
            rewards.append(rewards_this)
        
        return torch.LongTensor(indices), candidates, torch.FloatTensor(rewards)

    
    def get_reward(self, targets,  preds):
        """
        args:
            targets: list of targets for each sample
            preds: list of predictions, length == len(targets) * num_cand
        returns:
            rewards: the scores
            NOTE: We should always keep the positive sequences in the first candidate for each sample
        """
        num_cand = len(preds)//len(targets)
        preds_processed, targets_processed = self.postprocess_text(preds, targets)

        rewards = []
        for i,t in enumerate(targets_processed):
            scores = []
            ps = preds_processed[i * num_cand: (i+1)*num_cand]
            for j,p in enumerate(ps):
                s = self.scorer.score(t, p)
                scores.append(s["rouge1"].fmeasure / 0.45 + s["rouge2"].fmeasure / 0.2 + s["rougeLsum"].fmeasure / 0.4)

            rewards += scores
        
        return torch.FloatTensor(rewards)



class compute_qg:
    def __init__(self):
        self.rouge_metric = load_metric('rouge')
        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types =  ["rougeL"], use_stemmer=True)
        self.bleu_scorer = load_metric('bleu')
        self.meteor_scorer = load_metric('meteor')

    def postprocess_text_bleu(self, preds, labels):
        preds = [nltk.word_tokenize(clean(pred)) for pred in preds]
        labels = [nltk.word_tokenize(clean(label)) for label in labels]
        
        return preds, labels

    def __call__(self, eval_preds):
        preds, labels = eval_preds
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


    def get_candidates(self, targets,  preds, num_cand, max_num, strategy):
        """
        args:
            targets: list of targets for each sample
            pos: list of positive samples for each sample
            preds: list of predictions, length == len(targets) * num_cand
            num_cand: number of candidates
            max_num: number of returned indices per sample 
        returns:
            indices: Torch tensor, (B * (C-1), ), since the positive candidate is not generated from the generator
            candiates: candidates, with the length of len(targets) * max_num
            NOTE: We should always keep the positive sequences in the first candidate for each sample
        """
        preds_bleu, targets_bleu = self.postprocess_text_bleu(preds, targets)
        # preds_meteor = [' '.join(pred) for pred in preds_bleu]
        # targets_meteor = [' '.join(label) for label in targets_bleu]
        # print(targets_meteor)

        indices = []
        candidates = []
        rewards = []
        for i,t in enumerate(targets):
            scores = []
            ps = preds[i * num_cand: (i+1)*num_cand]
            ps_bleu = preds_bleu[i * num_cand: (i+1)*num_cand]
            for j,p in enumerate(ps):
                if len(ps_bleu[j]) == 0:
                    rouge_score = 0
                    bleu_score = 0
                    meteor_score = 0
                else:
                    rouge_score = self.rouge_scorer.score(t, p)['rougeL'].fmeasure
                    bleu_score = self.bleu_scorer._compute([ps_bleu[j]], [[targets_bleu[i]]], max_order = 4)['bleu']
                    meteor_score = self.meteor_scorer._compute([ps_bleu[j]], [targets_bleu[i]])['meteor']
                scores.append((j + i * num_cand, rouge_score / 0.5 + bleu_score/0.23 + meteor_score/0.27, p))

            scores = sorted(scores, key = lambda x: x[1], reverse=True)
            

            idx_this = [scores[0][0]] # the first as pos
            cand_this = [scores[0][2]]
            rewards_this = [scores[0][1]]
            scores = scores[1:]

            if strategy == 'random':
                s_for_pick = random.sample(scores, max_num - 1)
                idx_this +=  [s[0] for s in s_for_pick]
                cand_this +=  [s[2] for s in s_for_pick]
                rewards_this += [s[1] for s in s_for_pick]
            else:
                if strategy == 'top':
                    idx_this +=  [s[0] for s in scores[:max_num-1]]
                    cand_this +=  [s[2] for s in scores[:max_num-1]]
                    rewards_this += [s[1] for s in scores[:max_num-1]]
                elif strategy == 'bottom':
                    idx_this +=  [s[0] for s in scores[-max_num+1:]]
                    cand_this +=  [s[2] for s in scores[-max_num+1:]]
                    rewards_this += [s[1] for s in scores[-max_num+1:]]
                elif strategy == 'top-bottom':
                    n_top = (max_num-1) // 2
                    n_bottom = (max_num-1) - n_top
                    idx_this +=  [s[0] for s in scores[:n_top]]
                    cand_this += [s[2] for s in scores[:n_top]]
                    idx_this +=  [s[0] for s in scores[-n_bottom:]]
                    cand_this += [s[2] for s in scores[-n_bottom:]]
                    rewards_this += [s[1] for s in scores[:n_top]]
                    rewards_this += [s[1] for s in scores[-n_bottom:]]


            indices += idx_this
            candidates += cand_this
            rewards.append(rewards_this)
        
        return torch.LongTensor(indices), candidates, torch.FloatTensor(rewards)

    
    def get_reward(self, targets,  preds):
        """
        args:
            targets: list of targets for each sample
            preds: list of predictions, length == len(targets) * num_cand
        returns:
            rewards: the scores
            NOTE: We should always keep the positive sequences in the first candidate for each sample
        """
        num_cand = len(preds)//len(targets)
        preds_bleu, targets_bleu = self.postprocess_text_bleu(preds, targets)
        preds_meteor = [' '.join(pred) for pred in preds_bleu]
        targets_meteor = [' '.join(label) for label in targets_bleu]

        rewards = []
        for i,t in enumerate(targets):
            scores = []
            ps = preds[i * num_cand: (i+1)*num_cand]
            ps_bleu = preds_bleu[i * num_cand: (i+1)*num_cand]
            ps_meteor = preds_meteor[i * num_cand: (i+1)*num_cand]
            for j,p in enumerate(ps):
                rouge_score = self.rouge_scorer.score(t, p)['rougeL']
                bleu_score = self.bleu_scorer._compute([ps_bleu[j]], [[targets_bleu[i]]], max_order = 4)['bleu']
                meteor_score = self.bleu_scorer._compute([ps_meteor[j]], [targets_meteor[i]])['meteor']
                scores.append(rouge_score / 0.5 + bleu_score/0.23 + meteor_score/0.27)

            rewards += scores
        
        return torch.FloatTensor(rewards)



class compute_dialog:
    def __init__(self):
        self.bleu_scorer = load_metric('bleu')

    def postprocess_text_bleu(self, preds, labels):
        preds = [nltk.word_tokenize(clean(pred)) for pred in preds]
        labels = [nltk.word_tokenize(clean(label)) for label in labels]

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

    def __call__(self, eval_preds):
        preds, labels = eval_preds

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


    def get_candidates(self, targets,  preds, num_cand, max_num, strategy):
        """
        args:
            targets: list of targets for each sample
            pos: list of positive samples for each sample
            preds: list of predictions, length == len(targets) * num_cand
            num_cand: number of candidates
            max_num: number of returned indices per sample 
        returns:
            indices: Torch tensor, (B * (C-1), ), since the positive candidate is not generated from the generator
            candiates: candidates, with the length of len(targets) * max_num
            NOTE: We should always keep the positive sequences in the first candidate for each sample
        """
        preds_bleu, targets_bleu = self.postprocess_text_bleu(preds, targets)

        indices = []
        candidates = []
        rewards = []
        for i,t in enumerate(targets):
            scores = []
            ps = preds[i * num_cand: (i+1)*num_cand]
            ps_bleu = preds_bleu[i * num_cand: (i+1)*num_cand]
            for j,p in enumerate(ps):
                if len(ps_bleu[j]) == 0:
                    bleu_score_1 = 0
                    bleu_score_2 = 0
                else:
                    bleu_score_1, bleu_score_2 = self.bleu([ps_bleu[j]], [targets_bleu[i]])
                _,_, d1, d2 = self.distinct([ps_bleu[j]])
                scores.append((j + i * num_cand, bleu_score_1 / 0.5 + bleu_score_2 / 0.4 + d1/2 + d2/2, p))

            scores = sorted(scores, key = lambda x: x[1], reverse=True)
            

            idx_this = [scores[0][0]] # the first as pos
            cand_this = [scores[0][2]]
            rewards_this = [scores[0][1]]
            scores = scores[1:]

            if strategy == 'random':
                s_for_pick = random.sample(scores, max_num - 1)
                idx_this +=  [s[0] for s in s_for_pick]
                cand_this +=  [s[2] for s in s_for_pick]
                rewards_this += [s[1] for s in s_for_pick]
            else:
                if strategy == 'top':
                    idx_this +=  [s[0] for s in scores[:max_num-1]]
                    cand_this +=  [s[2] for s in scores[:max_num-1]]
                    rewards_this += [s[1] for s in scores[:max_num-1]]
                elif strategy == 'bottom':
                    idx_this +=  [s[0] for s in scores[-max_num+1:]]
                    cand_this +=  [s[2] for s in scores[-max_num+1:]]
                    rewards_this += [s[1] for s in scores[-max_num+1:]]
                elif strategy == 'top-bottom':
                    n_top = (max_num-1) // 2
                    n_bottom = (max_num-1) - n_top
                    idx_this +=  [s[0] for s in scores[:n_top]]
                    cand_this += [s[2] for s in scores[:n_top]]
                    idx_this +=  [s[0] for s in scores[-n_bottom:]]
                    cand_this += [s[2] for s in scores[-n_bottom:]]
                    rewards_this += [s[1] for s in scores[:n_top]]
                    rewards_this += [s[1] for s in scores[-n_bottom:]]


            indices += idx_this
            candidates += cand_this
            rewards.append(rewards_this)
        
        return torch.LongTensor(indices), candidates, torch.FloatTensor(rewards)

    
    def get_reward(self, targets,  preds):
        """
        args:
            targets: list of targets for each sample
            preds: list of predictions, length == len(targets) * num_cand
        returns:
            rewards: the scores
            NOTE: We should always keep the positive sequences in the first candidate for each sample
        """
        num_cand = len(preds)//len(targets)
        preds_bleu, targets_bleu = self.postprocess_text_bleu(preds, targets)

        rewards = []
        for i,t in enumerate(targets):
            scores = []
            ps = preds[i * num_cand: (i+1)*num_cand]
            ps_bleu = preds_bleu[i * num_cand: (i+1)*num_cand]
            for j,p in enumerate(ps):
                bleu_score_1, bleu_score_2 = self.bleu([ps_bleu[j]], [targets_bleu[i]])
                # _,_, d1, d2 = self.distinct([ps_bleu[j]])
                scores.append( bleu_score_1 / 0.5 + bleu_score_2 / 0.4)

            rewards += scores
        
        return torch.FloatTensor(rewards)





class compute_coqa:
    def __init__(self):
        pass

    def postprocess_text_coqa(self, preds, labels):
        preds = [' '.join(nltk.word_tokenize(clean(pred))) for pred in preds]
        labels = [' '.join(nltk.word_tokenize(clean(label))) for label in labels]

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

    def __call__(self, eval_preds):
        preds, labels = eval_preds

        # Some simple post-processing
        preds_coqa, labels_coqa = self.postprocess_text_coqa(preds, labels)

        # Extract a few results from ROUGE
        result = {}
        result['f1'] = CoQAEvaluator.quick_model_performance(preds_coqa, labels_coqa) * 100

        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    def get_candidates(self, targets,  preds, num_cand, max_num, strategy):
        """
        args:
            targets: list of targets for each sample
            pos: list of positive samples for each sample
            preds: list of predictions, length == len(targets) * num_cand
            num_cand: number of candidates
            max_num: number of returned indices per sample 
        returns:
            indices: Torch tensor, (B * (C-1), ), since the positive candidate is not generated from the generator
            candiates: candidates, with the length of len(targets) * max_num
            NOTE: We should always keep the positive sequences in the first candidate for each sample
        """
        preds_coqa, targets_coqa = self.postprocess_text_coqa(preds, targets)

        indices = []
        candidates = []
        rewards = []
        for i,t in enumerate(targets):
            scores = []
            ps = preds[i * num_cand: (i+1)*num_cand]
            ps_coqa = preds_coqa[i * num_cand: (i+1)*num_cand]
            for j,p in enumerate(ps):
                f1 = CoQAEvaluator.compute_f1(ps_coqa[j], targets_coqa[i])
                # _,_, d1, d2 = self.distinct([ps_bleu[j]])
                scores.append((j + i * num_cand, f1, p))

            scores = sorted(scores, key = lambda x: x[1], reverse=True)
            

            idx_this = [scores[0][0]] # the first as pos
            cand_this = [scores[0][2]]
            rewards_this = [scores[0][1]]
            scores = scores[1:]

            if strategy == 'random':
                s_for_pick = random.sample(scores, max_num - 1)
                idx_this +=  [s[0] for s in s_for_pick]
                cand_this +=  [s[2] for s in s_for_pick]
                rewards_this += [s[1] for s in s_for_pick]
            else:
                if strategy == 'top':
                    idx_this +=  [s[0] for s in scores[:max_num-1]]
                    cand_this +=  [s[2] for s in scores[:max_num-1]]
                    rewards_this += [s[1] for s in scores[:max_num-1]]
                elif strategy == 'bottom':
                    idx_this +=  [s[0] for s in scores[-max_num+1:]]
                    cand_this +=  [s[2] for s in scores[-max_num+1:]]
                    rewards_this += [s[1] for s in scores[-max_num+1:]]
                elif strategy == 'top-bottom':
                    n_top = (max_num-1) // 2
                    n_bottom = (max_num-1) - n_top
                    idx_this +=  [s[0] for s in scores[:n_top]]
                    cand_this += [s[2] for s in scores[:n_top]]
                    idx_this +=  [s[0] for s in scores[-n_bottom:]]
                    cand_this += [s[2] for s in scores[-n_bottom:]]
                    rewards_this += [s[1] for s in scores[:n_top]]
                    rewards_this += [s[1] for s in scores[-n_bottom:]]


            indices += idx_this
            candidates += cand_this
            rewards.append(rewards_this)
        
        return torch.LongTensor(indices), candidates, torch.FloatTensor(rewards)

    
    def get_reward(self, targets,  preds):
        """
        args:
            targets: list of targets for each sample
            preds: list of predictions, length == len(targets) * num_cand
        returns:
            rewards: the scores
            NOTE: We should always keep the positive sequences in the first candidate for each sample
        """
        num_cand = len(preds)//len(targets)
        preds_coqa, targets_coqa = self.postprocess_text_coqa(preds, targets)

        rewards = []
        for i,t in enumerate(targets):
            scores = []
            ps = preds[i * num_cand: (i+1)*num_cand]
            ps_coqa = preds_coqa[i * num_cand: (i+1)*num_cand]
            for j,p in enumerate(ps):
                f1 = CoQAEvaluator.compute_f1(ps_coqa[j], targets_coqa[i])
                # _,_, d1, d2 = self.distinct([ps_bleu[j]])
                scores.append(f1)

            rewards += scores
        
        return torch.FloatTensor(rewards)