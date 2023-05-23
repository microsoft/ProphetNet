import json
import os
import argparse
from os import write
from tqdm import tqdm
import queue
import numpy as np
from rouge_score import rouge_scorer, scoring
from transformers import BartTokenizer, RobertaTokenizer, PreTrainedTokenizerFast, BartTokenizerFast, RobertaTokenizerFast
import random
import pickle
import time
import nltk

nltk.download('punkt')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data_dir", type=str, default='./')
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--tokenizer_dir", type=str, default='bart-base')
#parser.add_argument("--golden", type=str, help="Gold output file.")
args = parser.parse_args()


def read(dataset_name, raw_data_dir, generator_tokenizer, split):
    source_texts = []
    with open(os.path.join(raw_data_dir, dataset_name, 'org_data/%s.src'%( split)), encoding='utf-8') as f:
        for line in f.readlines():
            source_texts.append(line.strip())
    
    target_texts = []
    with open(os.path.join(raw_data_dir,dataset_name,  'org_data/%s.tgt'%( split)), encoding='utf-8') as f:
        for line in f.readlines():
            target_texts.append(line.strip())
    
    # rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    # scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
    # candidates = []
    # with open(candidate_dir + '/%s/generated_predictions_%s.txt'%(split, num_cand), encoding='utf-8') as f:
    #     for line in f.readlines():
    #         candidates.append(line.strip())  
    
    cur_line = 0
    samples = []
    i = 0
    article_tokens = []
    target_tokens = []
    zip_seqs = []
    for s, t in zip(source_texts, target_texts):
        zip_seqs.append((s, t))

    for k in tqdm(zip_seqs):
        # s = s.replace('[SEP]', '</s>')
        # t = t.replace('[SEP]', '</s>')
        s = k[0]
        t = k[1]

        # since the glge dataset is tokenized, we recover them to the normal text
        article_tokens.append(generator_tokenizer.convert_tokens_to_ids(generator_tokenizer.tokenize(s)))
        target_tokens.append(generator_tokenizer.convert_tokens_to_ids(generator_tokenizer.tokenize(t)))
        # target_for_score = '\n'.join(sent_detector.tokenize(target))
        # cand_list = []
        # for _ in range(num_cand):
        #     # cand = candidates[cur_line]
        #     # cand_for_score = '\n'.join(sent_detector.tokenize(cand))
        #     score = scorer.score(target_for_score, cand_for_score)
        #     # similarity = (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
        #     similarity = score["rouge1"].fmeasure / 0.45 + score["rouge2"].fmeasure / 0.2 + score["rougeLsum"].fmeasure / 0.4
        #     cand_list.append((cand, similarity))
        #     cur_line += 1
        
        # cand_list = sorted(cand_list, key=lambda x:x[1], reverse=True)
        # cand_list = [c[0] for c in cand_list]

    article_texts = generator_tokenizer.batch_decode(article_tokens, skip_special_tokens=True)
    target_texts = generator_tokenizer.batch_decode(target_tokens, skip_special_tokens=True)

    samples = []
    for s,t in zip(article_texts, target_texts):
        samples.append({
            'source': s,
            'target':t,
            # 'oracle':cand_list[0],
            # 'candidates': cand_list
        })

    return samples


def process_and_write(dataset_name, raw_data_dir, split, generator_tokenizer):
    print('processing %s set...'%(split))
    
    process_start = time.time()
    train_set = read(dataset_name, raw_data_dir, generator_tokenizer, split)
    process_end = time.time()

    print('processing %s set cost'%(split), process_end-process_start,'s')

    print('saving %s set json files...'%(split))
    with open('%s/%s_data.json'%(dataset_name, split), 'w', encoding='utf-8') as f:
        json.dump(train_set, f)
    save_json_end = time.time()
    print('saving %s set json files cost'%(split), save_json_end-process_end,'s')



if __name__ == "__main__":
    generator_tokenizer = BartTokenizerFast.from_pretrained(args.tokenizer_dir)
    candidate_dir = None
    
    if not os.path.exists(args.dataset_name):
        os.makedirs(args.dataset_name)

    process_and_write(args.dataset_name, args.raw_data_dir, 'train', generator_tokenizer=generator_tokenizer)
    process_and_write(args.dataset_name, args.raw_data_dir, 'dev', generator_tokenizer=generator_tokenizer)
    process_and_write(args.dataset_name, args.raw_data_dir, 'test', generator_tokenizer=generator_tokenizer)
    # process_and_write('train_1',tokenizer=tokenizer,  max_source_length=max_source_length, max_target_length = max_target_length)
    # process_and_write('train_2',tokenizer=tokenizer,  max_source_length=max_source_length, max_target_length = max_target_length)
