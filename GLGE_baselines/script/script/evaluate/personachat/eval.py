from collections import Counter

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
from argparse import ArgumentParser

def distinct(seqs):
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


def bleu(hyps, refs):
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--golden-file', dest="golden_file", help='Input data file, one golden per line.')
    parser.add_argument('--pred-file', dest="pred_file", help='Model predictions.')
    args = parser.parse_args()
    
    with open(args.pred_file, encoding='utf-8') as fin:
        preds = fin.readlines()
        preds = [line.strip().split(" ") for line in preds]
    with open(args.golden_file, encoding='utf-8') as fin:
        golds = fin.readlines()
        golds = [line.strip().split(" ") for line in golds]
    
   
    bleu1, bleu2 = bleu(preds, golds)
    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(preds)
    print(bleu1 * 100., bleu2 * 100., inter_dist1, inter_dist2)

