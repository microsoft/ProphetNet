import os
import sys
import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

input_file = sys.argv[1]
output_file = sys.argv[2]

tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')

with open(input_file, 'r', encoding='utf-8') as fin:
    fout = open('{}'.format(output_file), 'w', encoding='utf-8')
    for line in fin:
        line = line.strip().split(" ")
        line = tokenizer.convert_tokens_to_string(line)
        fout.write("{}\n".format(line))
