import argparse
import os
from os import listdir
from os.path import isfile, join
import re
import subprocess

from nltk.tokenize.treebank import TreebankWordDetokenizer
import tqdm
from pytorch_transformers import BertTokenizer

def uncased_preocess(fin, fout, keep_sep=False, max_len=512):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    twd = TreebankWordDetokenizer()
    bpe = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in tqdm.tqdm(fin.readlines()):
        line = line.strip().replace('``', '"').replace('\'\'', '"').replace('`', '\'')
        s_list = [twd.detokenize(x.strip().split(
            ' '), convert_parentheses=True) for x in line.split('<S_SEP>')]
        tk_list = [bpe.tokenize(s) for s in s_list]
        output_string_list = [" ".join(s) for s in tk_list]
        if keep_sep:
            output_string = " [X_SEP] ".join(output_string_list)
        else:
            output_string = " ".join(output_string_list)
        output_string = " ".join(output_string.split(' ')[:max_len-1])
        fout.write('{}\n'.format(output_string))
        
def tokenize_with_bert_uncase(fin, fout):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in tqdm.tqdm(fin.readlines()):
        new = tok.tokenize(line.strip())
        new_line = " ".join(new)
        fout.write('{}\n'.format(new_line))


def tokenize_data(version, dataset):
    input_dir = '../data/%s/%s_data/org_data' % (version, dataset)
    output_dir = '../data/%s/%s_data/uncased_tok_data' % (version, dataset)
    if dataset == 'cnndm':
        uncased_preocess('%s/train.src' % input_dir , '%s/train.src' % output_dir, keep_sep=False)
        uncased_preocess('%s/dev.src' % input_dir , '%s/dev.src' % output_dir, keep_sep=False)
        uncased_preocess('%s/test.src' % input_dir , '%s/test.src' % output_dir, keep_sep=False)
        uncased_preocess('%s/train.tgt' % input_dir , '%s/train.tgt' % output_dir, keep_sep=True, max_len=128)
        uncased_preocess('%s/dev.tgt' % input_dir , '%s/dev.tgt' % output_dir, keep_sep=True)
        uncased_preocess('%s/test.tgt' % input_dir , '%s/test.tgt' % output_dir, keep_sep=True)
    else:
        tokenize_with_bert_uncase('%s/train.src' % input_dir , '%s/train.src' % output_dir)
        tokenize_with_bert_uncase('%s/train.tgt' % input_dir , '%s/train.tgt' % output_dir)
        tokenize_with_bert_uncase('%s/dev.src' % input_dir , '%s/dev.src' % output_dir)
        tokenize_with_bert_uncase('%s/dev.tgt' % input_dir , '%s/dev.tgt' % output_dir)
        tokenize_with_bert_uncase('%s/test.src' % input_dir , '%s/test.src' % output_dir)
        tokenize_with_bert_uncase('%s/test.tgt' % input_dir , '%s/test.tgt' % output_dir)
        
parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, help="choose data version from all, or 1 of 4 versions: easy, medium, medium+, hard")
parser.add_argument("--dataset", type=str, help="choose dataset from all, or 1 of 8 datasets: cnndm, gigaword, xsum, msnews, msqg, squadqg, coqa, personachat")
args = parser.parse_args()

VERSION_LIST = ['easy', 'medium', 'medium+', 'hard']
DATASET_LIST = ['cnndm', 'gigaword', 'xsum', 'msnews', 'msqg', 'squadqg', 'coqa', 'personachat']

if args.version != 'all' and args.version not in VERSION_LIST:
    print('please choose version from all, or 1 of 4 versions: easy, medium, medium+, hard')
    exit()
else:
    if args.version == 'all':
        version_list = VERSION_LIST
    else:
        version_list = [args.version]
    
if args.dataset != 'all' and args.dataset not in DATASET_LIST:
    print('please choose dataset from all, or 1 of 8 datasets: cnndm, gigaword, xsum, msnews, msqg, squadqg, coqa, personachat')
    exit()
else:
    if args.dataset == 'all':
        dataset_list = DATASET_LIST
    else:
        dataset_list = [args.dataset]
        
print(version_list, dataset_list)
for dataset in dataset_list:
    for version in version_list:
        tokenize_data(version, dataset)