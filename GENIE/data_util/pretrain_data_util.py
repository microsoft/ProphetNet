from tqdm import tqdm
import os
from transformers import AutoTokenizer
from torch.utils.data.dataset import Dataset
import torch
import jsonlines
import numpy as np
import random


def load_loop_pretrain_data(args, padding_mode, tokenizer, data_name = None):
    print("***** load " + data_name + " train src dataset*****")

    path = os.path.join(args.data_path, data_name + '.npy')
    input_id_list = np.load(path, allow_pickle=True)

    # filter
    input_id_list = np.array([input_id for input_id in input_id_list if np.count_nonzero(input_id) >= 30])

    if padding_mode == 'max_len':
        dataset = Pre_dataset(input_id_list, tokenizer, mask_pro=args.mask_pro, maxlength=args.pre_max_len)
    elif padding_mode == 'conti_tgt':
        print("using new pretrain method...")
        dataset = Pre_dataset_type2(input_id_list, tokenizer, mask_pro=args.mask_pro, maxlength=args.pre_max_len)
    elif padding_mode == 'block':
        print("padding block is under realization")
        pass
    else:
        return NotImplementedError

    print("example of src id lists: ", dataset[50][0])
    print("example of tgt id lists: ", dataset[50][1])
    print("total query dataset len :", len(dataset))

    return dataset



def load_pretrain_data(args, padding_mode, tokenizer, data_name = None):
    questions = []
    print("***** load " + data_name + " train src dataset*****")

    input_id_list = None

    if data_name == "book" or data_name == "openweb" or data_name == "wiki" or data_name == "stories":
        for i in range(5):
            path = os.path.join(args.data_path, args.data_name + str(i+1) + '.npy')
            input_id_list_pre = np.load(path, allow_pickle=True)
            if i == 0:
                input_id_list = input_id_list_pre
            else:
                input_id_list = np.concatenate((input_id_list, input_id_list_pre), axis=0)
            # with open(path, "r", encoding="utf-8") as ifile:
            #     for line in tqdm(ifile):
            #         line = line.strip()
            #         text = line
            #         tgt.append(text)

    elif data_name == 'realnews':
        # for i in range(10):
        #     path = os.path.join(args.data_path, args.data_name + str(i+1) + '.txt')
        #     with open(path, "r", encoding="utf-8") as ifile:
        #         for line in tqdm(ifile):
        #             line = line.strip()
        #             text = line
        #             tgt.append(text)
        for i in range(10):
            path = os.path.join(args.data_path, args.data_name + str(i+1) + '.npy')
            input_id_list_pre = np.load(path, allow_pickle=True)
            if i == 0:
                input_id_list = input_id_list_pre
            else:
                input_id_list = np.concatenate((input_id_list, input_id_list_pre), axis=0)

    else:
        return NotImplementedError

    # filter
    input_id_list = np.array([input_id for input_id in input_id_list if np.count_nonzero(input_id) >= 256])


    # print("example of src text: ", src[50])
    print("example of input id: ", input_id_list[50])

    if padding_mode == 'max_len':
        dataset = Pre_dataset(input_id_list, tokenizer, mask_pro=args.mask_pro, maxlength=args.pre_max_len)
    elif padding_mode == 'conti_tgt':
        print("using new pretrain method...")
        dataset = Pre_dataset_type2(input_id_list, tokenizer, mask_pro=args.mask_pro, maxlength=args.pre_max_len)
    elif padding_mode == 'block':
        print("padding block is under realization")
        pass
    else:
        return NotImplementedError

    print("example of src id lists: ", dataset[50][0])
    print("example of tgt id lists: ", dataset[50][1])
    print("total query dataset len :", len(dataset))

    return dataset



class Pre_dataset(Dataset):
    def __init__(self, tgt_id, tokenizer, mask_pro=0.3, maxlength=512, span_size=8, mask_mode='random'):
        self.tgt_id = tgt_id
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        self.mask_pro = mask_pro
        self.span_size = span_size
        self.mask_token_index = self.tokenizer.mask_token_id
        self.pad_token_index = self.tokenizer.pad_token_id
        self.all_special_token = self.tokenizer.all_special_ids


    def __getitem__(self, index):
        tgt_example = self.tgt_id[index]
        # src_input_ids = tgt_example.tolist()
        tgt_input_ids = (torch.from_numpy(tgt_example)).long()
        src_input_ids = tgt_input_ids.clone()
        id_len = torch.nonzero(src_input_ids).shape[0]
        mask_span_num = int((id_len * self.mask_pro) // self.span_size) + 1
        # print("mask_span_num:", mask_span_num)
        mask_index = torch.randint(0, id_len, (mask_span_num,))
        # print("mask_index:", mask_index)
        mask_id_mask = torch.full(src_input_ids.shape, False, dtype=torch.bool)
        retain_id_mask = torch.full(src_input_ids.shape, True, dtype=torch.bool)
        mask_id_mask[mask_index] = True

        del_index = mask_index.tolist()
        for i in mask_index:
            del_index.extend(list(range(i + 1, i + self.span_size)))
        del_index = [i for i in del_index if i < id_len]
        del_index = torch.from_numpy(np.array(list(set(del_index))))
        # print("del_index", del_index)
        retain_id_mask[del_index] = False
        retain_id_mask = retain_id_mask | mask_id_mask
        src_input_ids[mask_id_mask] = self.mask_token_index
        src_input_ids = src_input_ids[retain_id_mask].tolist()
        # print("src_input_ids1:", len(src_input_ids))
        src_input_ids = src_input_ids + [self.pad_token_index] * (self.maxlength - len(src_input_ids))
        # print("src_input_ids2:", len(src_input_ids))
        src_input_ids = torch.from_numpy(np.array(src_input_ids)).long()

        return src_input_ids.unsqueeze(0), tgt_input_ids.unsqueeze(0)

    def __len__(self):
        return len(self.tgt_id)

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            src_tensor = torch.cat([feature[0] for feature in features])
            tgt_tensor = torch.cat([feature[1] for feature in features])
            return { "src_input_ids": src_tensor, "src_attention_mask": (src_tensor != 0).long(),
                     "tgt_input_ids": tgt_tensor, "tgt_attention_mask": (tgt_tensor != 0).long() }

        return fn

class Pre_dataset_type2(Dataset):
    def __init__(self, tgt_id, tokenizer, mask_pro=0.3, maxlength=512, mask_mode='random'):
        self.tgt_id = tgt_id
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        self.mask_pro = mask_pro
        self.tgtmaxlength = int(maxlength * mask_pro) + 1
        self.mask_token_index = self.tokenizer.mask_token_id
        self.pad_token_index = self.tokenizer.pad_token_id
        self.all_special_token = self.tokenizer.all_special_ids


    def __getitem__(self, index):
        tgt_example = self.tgt_id[index]
        # src_input_ids = tgt_example.tolist()
        tgt_input_ids = (torch.from_numpy(tgt_example)).long()
        src_input_ids = tgt_input_ids.clone()
        id_len = torch.nonzero(src_input_ids).shape[0]

        # mask_span_num = int((id_len * self.mask_pro) // self.span_size) + 1
        mask_span_len = int(id_len * self.mask_pro)
        # print("mask_span_num:", mask_span_num)
        mask_index = random.randint(0, id_len-mask_span_len-1)

        tgt_input_ids = src_input_ids.tolist()[mask_index:mask_index+mask_span_len]

        src_input_ids[mask_index] = self.mask_token_index

        # print("mask_index:", mask_index)
        # mask_span_len
        # mask_id_mask = torch.full(src_input_ids.shape, False, dtype=torch.bool)
        retain_id_mask = torch.full(src_input_ids.shape, True, dtype=torch.bool)
        # mask_id_mask[mask_index] = True

        # del_index = mask_index.tolist()
        del_index = list(range(mask_index + 1, mask_index + mask_span_len))
        del_index = torch.from_numpy(np.array(del_index))
        retain_id_mask[del_index] = False
        # src_input_ids[mask_id_mask] = self.mask_token_index
        src_input_ids = src_input_ids[retain_id_mask].tolist()
        # print("src_input_ids1:", len(src_input_ids))
        src_input_ids = src_input_ids + [self.pad_token_index] * (self.maxlength - len(src_input_ids))
        # print("src_input_ids2:", len(src_input_ids))
        tgt_input_ids = tgt_input_ids + [self.pad_token_index] * (self.tgtmaxlength - len(tgt_input_ids))

        src_input_ids = torch.from_numpy(np.array(src_input_ids)).long()
        tgt_input_ids = torch.from_numpy(np.array(tgt_input_ids)).long()

        return src_input_ids.unsqueeze(0), tgt_input_ids.unsqueeze(0)

    def __len__(self):
        return len(self.tgt_id)

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            src_tensor = torch.cat([feature[0] for feature in features])
            tgt_tensor = torch.cat([feature[1] for feature in features])
            return { "src_input_ids": src_tensor, "src_attention_mask": (src_tensor != 0).long(),
                     "tgt_input_ids": tgt_tensor, "tgt_attention_mask": (tgt_tensor != 0).long() }

        return fn

if __name__ == "__main__":
    pretrain_max_len = 512