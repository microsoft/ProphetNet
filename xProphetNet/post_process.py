import os
import sys
import sentencepiece as spm

from nltk.translate.bleu_score import corpus_bleu

f_in = sys.argv[1]
f_out = sys.argv[2]

sp = spm.SentencePieceProcessor()
sp.Load("xlmr_dictionary/sentencepiece.bpe.model")

fin = open(f_in, 'r', encoding='utf-8')
fout = open(f_out, 'w', encoding='utf-8')
for line in fin:
		gen = line.strip().split()
		gen = sp.DecodePieces(gen)
		fout.write('{}\n'.format(gen))
