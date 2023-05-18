import argparse

from datasets import load_metric
from sacrebleu import corpus_bleu
from sacremoses import MosesDetokenizer, MosesTokenizer


parser = argparse.ArgumentParser()

parser.add_argument('--ref', type=str, default='')
parser.add_argument('--pred', type=str, default='')
parser.add_argument('--lang', type=str, default='')

args = parser.parse_args()


ref_data = open(args.ref, 'r', encoding='utf-8').readlines()
pred_data = open(args.pred, 'r', encoding='utf-8').readlines()
assert len(ref_data) == len(pred_data)

print('reference: ', ref_data[0])
print('generate: ', pred_data[0])


mt, md = MosesTokenizer(lang=args.lang), MosesDetokenizer(lang=args.lang)
metric_sacrebleu = load_metric("./eval_utils/iwslt_wmt/sacre_bleu.py")


refs = [[md.detokenize(mt.tokenize(item))] for item in ref_data]
preds = [md.detokenize(mt.tokenize(item)) for item in pred_data]
sacre_results = metric_sacrebleu.compute(predictions=preds, references=refs)
print('***seqdiffuseq sacrebleu', round(sacre_results['score'], 2))

tok = "13a" if args.lang == "en" else "intl"
refs = [md.detokenize(item.split()) for item in ref_data]
preds = [md.detokenize(item.split()) for item in pred_data]
print("***dinoiser sacrebleu: ", corpus_bleu(preds, [refs], tokenize=tok))
