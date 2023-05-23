from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from functools import partial
import time
from transformers import RobertaTokenizer
import random
import pickle
import copy
import logging
import math
import copy
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


def encode_seq(tokenizer, seq):
    """
    encode the tokenized or untokenized sequence to token ids
    """
    # print(seq)
    if seq == [] or seq == "":
        return []
    else:
        return tokenizer.encode(seq, add_special_tokens=False)


class ReRankingDataset_eval(Dataset):
    def __init__(self, fdir, tokenizer, split = "dev", args = None, is_test=False):
        """ data format: article, abstract, [(candidiate_i, score_i)] 
        args:
            fdir: data dir or dataset name
            split 
            tokenizer
            args: data argument
            num_cand: number of generated candidates
            maxlen: max length for candidates
            is_test
            total_len: total input length
            is_sorted: sort the candidates by similarity
            max_num: max number of candidates in the input sample
            use_untok: use the untokenized data (used when the preprocessed data is tokenized by unsuitable tokenizer)
            cache_data: whether to cache data in memory, useful when training on cloud with blob container
        """
        self.isdir = os.path.isdir(fdir)
        self.args = args
        self.is_test = is_test
        if self.isdir:
            # directly input the data file dir
            self.fdir = fdir
        else:
            # input dataset name
            self.fdir = "data/%s/%s/gen_from_%d"%(fdir, split, self.args.num_cand)

        # to speed up the data loading in remote server, we minimize the data loading operation
        # self.num = len(os.listdir(self.fdir))
        # if os.path.exists(os.path.join(self.fdir, "all.json")):
        #     self.num = self.num - 1
        if not os.path.isfile(os.path.join(self.fdir, "all.json")) or not self.args.cache_data:
            self.num = len(os.listdir(self.fdir))
            if not os.path.isfile(os.path.join(self.fdir, "all.json")):
                self.num = self.num - 1

        self.tok = tokenizer
        self.pad_token_id = self.tok.pad_token_id
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id

        if self.args.cache_data:
            self.data = self.read_data()

    def _extract_sample(self, data):
        '''
            extra data from a raw data file
        '''
        if self.args.use_untokenized_data:
            article = data["source_untok"]
        else:
            article = data["source"]
        src_input_ids = encode_seq(self.tok, article)
        # if self.use_untok:
        #     abstract = data["target_untok"]
        # else: 
        #     abstract = data["target"]
        # tgt_input_ids = self.tok.encode(abstract, add_special_tokens=False)

        candidates = data["candidates"]
        _candidates = data["candidates_untok"]
        data["candidates"] = _candidates
        if self.args.use_untokenized_data:
            candidates = _candidates
        candidates = [encode_seq(self.tok, x) for x in candidates] # only trunk for candidates

        result = {
            "src_input_ids": src_input_ids, 
            # "tgt_input_ids": tgt_input_ids,
            "candidates": candidates
            }
        if self.is_test:
            result["data"] = {
                "source": data['source_untok'],
                "target": data['target_untok'],
                "candidates": data['candidates_untok']
            }
        
        return result


    def get_sample(self, idx):
        '''
            get one sample from one data file
        '''

        with open(os.path.join(self.fdir, "%d.json"%idx), "r") as f:
            data = json.load(f)

        return self._extract_sample(data)


    def read_data(self):
        '''
            cache the data in memory
        '''
        if os.path.isfile(os.path.join(self.fdir, "all.json")):
            with open(os.path.join(self.fdir, "all.json"), 'r') as f:
                raw_data = json.load(f)
            self.num = len(raw_data)
            data = []
            for i in range(self.num):
                data.append(self._extract_sample(raw_data[i]))
        else:  
            data = []
            for i in range(self.num):
                data.append(self.get_sample(i))
        return data

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.args.cache_data:
            sample = self.data[idx]
        else:
            sample = self.get_sample(idx)

        # cands = random.sample(data['candidates']) # random picked samples this is for training set
        # cands = sorted(cands, key=lambda x: x[1], reverse=True)
        cands = sample['candidates']
        
        input_ids = []
        for c in cands:
            input_id = [self.tok.cls_token_id]
            input_id += sample['src_input_ids'][:self.args.max_source_length]
            input_id += [self.tok.sep_token_id]
            input_id += c[:self.args.max_candidate_length]
            input_ids.append(input_id) 
        # padding for input id is left for collate_fn
        
        ret = {"input_ids": input_ids}
        if self.is_test:
            ret['data'] = sample['data']
        return ret




class ReRankingDataset_train(Dataset):
    def __init__(self, fdir, tokenizer, split = "train", args = None):
        """ data format: article, abstract, [(candidiate_i, score_i)] 
        args:
            fdir: data dir or dataset name
            split
            tokenizer
            args: data argument
            num_cand: number of generated candidates
            maxlen: max length for candidates
            is_test
            total_len: total input length
            is_sorted: sort the candidates by similarity
            max_num: max number of candidates in the input sample
            use_untok: use the untokenized data (used when the preprocessed data is tokenized by unsuitable tokenizer)
            cache_data: whether to cache data in memory, useful when training on cloud with blob container
        """
        self.isdir = os.path.isdir(fdir)
        self.args = args
        if self.isdir:
            # directly input the data file dir
            self.fdir = fdir
        else:
            # input dataset name
            self.fdir = "data/%s/%s/gen_from_%d"%(fdir, split, self.args.num_cand)

        # to speed up the data loading in remote server, we minimize the data loading operation
        # self.num = len(os.listdir(self.fdir))
        # if os.path.exists(os.path.join(self.fdir, "all.json")):
        #     self.num = self.num - 1
        if not os.path.isfile(os.path.join(self.fdir, "all.json")) or not self.args.cache_data:
            self.num = len(os.listdir(self.fdir))
            if not os.path.isfile(os.path.join(self.fdir, "all.json")):
                self.num = self.num - 1


        self.tok = tokenizer
        self.pad_token_id = self.tok.pad_token_id
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id

        if self.args.cache_data:
            self.data = self.read_data()

    def _extract_sample(self, data):
        '''
            extra data from a raw data file
        '''
       
        if self.args.use_untokenized_data:
            article = data["source_untok"]
        else:
            article = data["source"]
        src_input_ids = encode_seq(self.tok, article)
        if self.args.use_untokenized_data:
            abstract = data["target_untok"]
        else: 
            abstract = data["target"]
        tgt_input_ids = encode_seq(self.tok, abstract)

        candidates = data["candidates"]
        _candidates = data["candidates_untok"]
        
        data["candidates"] = _candidates
        if self.args.use_untokenized_data:
            candidates = _candidates
        candidates = [encode_seq(self.tok, x) for x in candidates] # only trunk for candidates
 
        result = {
            "src_input_ids": src_input_ids, 
            "tgt_input_ids": tgt_input_ids,
            "candidates": candidates

            }

        
        return result


    def get_sample(self, idx):
        '''
            get one sample from one data file
        '''

        with open(os.path.join(self.fdir, "%d.json"%idx), "r") as f:
            data = json.load(f)

        return self._extract_sample(data)


    def read_data(self):
        '''
            cache the data in memory
        '''
        if os.path.isfile(os.path.join(self.fdir, "all.json")):
            with open(os.path.join(self.fdir, "all.json"), 'r') as f:
                raw_data = json.load(f)
            self.num = len(raw_data)
            data = []
            for i in range(self.num):
                data.append(self._extract_sample(raw_data[i]))
        else:  
            data = []
            for i in range(self.num):
                data.append(self.get_sample(i))
        return data

    def __len__(self):
        return self.num


    def __getitem__(self, idx):
        if self.args.cache_data:
            sample = self.data[idx]
        else:
            sample = self.get_sample(idx)

        # # use the ground truth as pos
        # cands = [sample['tgt_input_ids']]
        # # the neg is always the worst candidate
        # # cands += [sample['candidates'][-1][0]]
        # # the neg is randomly sample from candidates
        # cands += random.sample([s[0] for s in sample['candidates']], self.args.max_num-1)

        
        # the pos is always the best candidate
        cands = [sample['candidates'][0]]
        # the neg is always the worst candidate
        cands += [c for c in sample['candidates'][-self.args.max_num+1:]]
        # cands += [c for c in sample['candidates'][1: self.args.max_num]]
        # # cands += random.sample([s[0] for s in sample['candidates'][1:]], self.args.max_num-1)

        # randomly pick two candidates
        # cands = random.sample(sample['candidates'], self.args.max_num) # random picked samples this is for training set
        # cands = sorted(cands, key=lambda x: x[1], reverse=True)
        # cands = [c[0] for c in cands]
        
        input_ids = []
        for c in cands:
            input_id = [self.tok.cls_token_id]
            input_id += sample['src_input_ids'][:self.args.max_source_length]
            input_id += [self.tok.sep_token_id]
            input_id += c[:self.args.max_candidate_length]
            input_ids.append(input_id) 
        # padding for input id is left for collate_fn
        
        ret = {"input_ids": input_ids}
        return ret
