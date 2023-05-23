import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import pandas as pd
import json
import copy
import numpy as np
import random
from pandas import DataFrame
import queue
import os
import pickle

class parsing_link:
    def __init__(self,num = -1,step = 0):
        self.num = num
        self.step = step

# 创建a与b有关
# deposite 是已经抛弃的长度
# def have_relation(relation,a,b, content_len, step):
#     relation[content_len[a]]
#     for i in range(content_len[a+1] - content_len[a]):
#         for j in range(content_len[b+1] - content_len[b]):
#             if i + content_len[a] - deposite >=0 and j + content_len[b] - deposite>=0:
#                 relation [i + content_len[a] - deposite] [j + content_len[b] - deposite] = 1


class SamsumDataset(Dataset):

    def __init__(self, dataset_name, split = 'train', tokenizer = None, args = None, shuffle=True):

        self.args = args
        self.data = self.read(dataset_name,split, tokenizer, shuffle)

        self.len = len(self.data)

    def read(self,dataset_name,split, tokenizer, shuffle = True):
        if os.path.exists('../data/%s/%s_data.pkl'%(dataset_name, split)) and self.args.use_tokenized_data:
            # print('load preprossed dataset from ../data/%s/%s_data.pkl'%(dataset_name, split))
            samples = pickle.load(open('../data/%s/%s_data.pkl'%(dataset_name, split),'rb'))
        else:
            with open('../data/%s/%s_data.json'%(dataset_name, split), encoding='utf-8') as f:
                raw_data = json.load(f)
            # process dialogue
            samples = []
            for d in raw_data:
                content_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<s> ' + d['source']))
                label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<s> '+d['target']))

                # if len(content_ids)>=self.args.max_sent_len:
                #     print(f'leng>{self.args.max_sent_len}')
                content_ids = content_ids[:self.args.max_source_length]
                # speaker_ids.insert(0,0)
                label = label[:self.args.max_target_length-1]
                label +=  tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</s>'))

                attention_mask = np.ones_like(content_ids)

                samples.append({
                    'content_ids': content_ids, #content_ids[self.args.max_sent_len:],
                    'labels': label,
                    'attention_mask': attention_mask
                })
        if split == 'train' and shuffle:
            random.shuffle(samples)
        return samples

    def __getitem__(self, index):
        '''

        :param index:
        :return:
            text_ids:
            token_types:
            label
        '''
        return torch.LongTensor(self.data[index]['content_ids']), \
               torch.LongTensor(self.data[index]['labels']) if 'labels' in self.data[index].keys() else torch.LongTensor(self.data[index]['target_ids']), \
               torch.FloatTensor(self.data[index]['attention_mask']) if 'attention_mask' in self.data[index].keys() else None

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        '''

        :param data:
            content_ids
            token_types
            labels

        :return:

        '''
        content_ids = pad_sequence([d[0] for d in data], batch_first = True, padding_value = 1) # (B, T, )
        labels = pad_sequence([d[1] for d in data], batch_first = True, padding_value=-100)


        if data[0][2] is not None:
            attention_mask = pad_sequence([d[2] for d in data], batch_first = True)
        else:
            attention_mask = None

        sample = {}
        sample['input_ids'] = content_ids
        sample['labels'] = labels
        sample['attention_mask'] = attention_mask
        # print(sample)
        # exit()
        return sample
