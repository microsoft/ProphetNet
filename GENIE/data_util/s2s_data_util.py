from tqdm import tqdm
import os
from transformers import AutoTokenizer
from torch.utils.data.dataset import Dataset
import torch
import jsonlines



def load_s2s_data(args, padding_mode, split, tokenizer):
    questions = []
    if split == 'train':
        print("***** load " + args.data_name + " train src dataset*****")
        src = []
        train_src_path = os.path.join(args.data_path, args.data_name + "/org_data/train.src")
        with open(train_src_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                src.append(text)

        print("***** load " + args.data_name + " train tgt dataset*****")
        tgt = []
        train_tgt_path = os.path.join(args.data_path, args.data_name + "/org_data/train.tgt")
        with open(train_tgt_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                tgt.append(text)

    elif split == 'dev':

        print("***** load " + args.data_name + " dev src dataset*****")
        src = []
        dev_src_path = os.path.join(args.data_path, args.data_name + "/org_data/dev.src")
        with open(dev_src_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                src.append(text)

        print("***** load " + args.data_name + " dev tgt dataset*****")
        tgt = []
        dev_tgt_path = os.path.join(args.data_path, args.data_name + "/org_data/dev.tgt")
        with open(dev_tgt_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                tgt.append(text)

    elif split == 'test':

        print("***** load " + args.data_name + " test src dataset*****")
        src = []
        test_src_path = os.path.join(args.data_path, args.data_name + "/org_data/test.src")
        with open(test_src_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                src.append(text)

        print("***** load " + args.data_name + " dev tgt dataset*****")
        tgt = []
        test_tgt_path = os.path.join(args.data_path, args.data_name + "/org_data/test.tgt")
        with open(test_tgt_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                tgt.append(text)

    else:
        print("no such split of data...")
        exit(0)

    print("example of src text: ", src[50])
    print("example of tgt text: ", tgt[50])

    if padding_mode == 'max_len':
        if args.data_name == "squadqg_data":
            dataset = QG_dataset_Diff(src, tgt, tokenizer, src_maxlength=args.src_max_len,
                                      answer_maxlength=args.answer_max_len, tgt_maxlength=args.tgt_max_len)
        else:
            dataset = S2S_dataset(src, tgt, tokenizer, src_maxlength=args.src_max_len, tgt_maxlength=args.tgt_max_len)
    elif padding_mode == 'block':
        print("padding block is under realization")
        pass
    else:
        return NotImplementedError

    print("example of src id lists: ", dataset[50][0])
    print("example of tgt id lists: ", dataset[50][1])
    print("total query dataset len :", len(dataset))

    return dataset


'''
for AR seq2seq training
'''
def load_s2s_data_AR(args, padding_mode, split, tokenizer):
    if split == 'train':
        print("***** load " + args.data_name + " train src dataset*****")
        src = []
        train_src_path = os.path.join(args.data_path, args.data_name + "/org_data/train.src")
        with open(train_src_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                src.append(text)

        print("***** load " + args.data_name + " train tgt dataset*****")
        tgt = []
        train_tgt_path = os.path.join(args.data_path, args.data_name + "/org_data/train.tgt")
        with open(train_tgt_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                tgt.append(text)

    elif split == 'dev':

        print("***** load " + args.data_name + " dev src dataset*****")
        src = []
        dev_src_path = os.path.join(args.data_path, args.data_name + "/org_data/dev.src")
        with open(dev_src_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                src.append(text)

        print("***** load " + args.data_name + " dev tgt dataset*****")
        tgt = []
        dev_tgt_path = os.path.join(args.data_path, args.data_name + "/org_data/dev.tgt")
        with open(dev_tgt_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                tgt.append(text)
        # src = src[:100]
        # tgt = tgt[:100]

    elif split == 'test':

        print("***** load " + args.data_name + " test src dataset*****")
        src = []
        test_src_path = os.path.join(args.data_path, args.data_name + "/org_data/test.src")
        with open(test_src_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                src.append(text)

        print("***** load " + args.data_name + " dev tgt dataset*****")
        tgt = []
        test_tgt_path = os.path.join(args.data_path, args.data_name + "/org_data/test.tgt")
        with open(test_tgt_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                tgt.append(text)
        # src = src[:10]
        # tgt = tgt[:10]

    else:
        print("no such split of data...")
        exit(0)

    print("example of src text: ", src[9].replace("<S_SEP>",'\n'))
    print("example of tgt text: ", tgt[9].replace("<S_SEP>",'\n'))

    if padding_mode == 'max_len':
        dataset = S2S_AR_dataset(src, tgt, tokenizer, src_maxlength=args.src_max_len, tgt_maxlength=args.tgt_max_len)
    elif padding_mode == 'block':
        print("padding block is under realization")
        pass
    else:
        return NotImplementedError

    print("example of src id lists: ", dataset[9][0])
    print("example of tgt id lists: ", dataset[9][1])
    print("total query dataset len :", len(dataset))

    return dataset

'''
load baseline data on SeqDiffusion
'''
def load_s2s_jsonl_data(args, padding_mode, split, tokenizer):
    questions = []
    if split == 'train':
        print("***** load " + args.data_name + " train src and tgt dataset*****")
        src = []
        tgt = []

        train_path = os.path.join(args.data_path, args.data_name + "/train.jsonl")
        with jsonlines.open(train_path) as reader:
            for obj in reader:
                tgt.append(obj['trg'])
                src.append(obj['src'])

    elif split == 'dev':

        print("***** load " + args.data_name + " dev src and tgt dataset*****")
        src = []
        tgt = []
        dev_path = os.path.join(args.data_path, args.data_name + "/valid.jsonl")
        with jsonlines.open(dev_path) as reader:
            for obj in reader:
                tgt.append(obj['trg'])
                src.append(obj['src'])

    elif split == 'test':

        print("***** load " + args.data_name + " test src and tgt dataset*****")
        src = []
        tgt = []
        test_path = os.path.join(args.data_path, args.data_name + "/test.jsonl")
        with jsonlines.open(test_path) as reader:
            for obj in reader:
                tgt.append(obj['trg'])
                src.append(obj['src'])

    else:
        print("no such split of data...")
        exit(0)

    print("example of src text: ", src[50])
    print("example of tgt text: ", tgt[50])

    if padding_mode == 'max_len':
        dataset = S2S_dataset(src, tgt, tokenizer, src_maxlength=args.src_max_len, tgt_maxlength=args.tgt_max_len)
    elif padding_mode == 'block':
        print("padding block is under realization")
        pass
    else:
        return NotImplementedError

    print("example of src id lists: ", dataset[50][0])
    print("example of tgt id lists: ", dataset[50][1])
    print("total query dataset len :", len(dataset))

    return dataset


class QG_dataset_Diff(Dataset):
    def __init__(self, src, tgt, tokenizer, src_maxlength=144, answer_maxlength=20, tgt_maxlength=32):
        self.src = src
        self.tgt = tgt
        self.tokenizer = tokenizer
        self.src_maxlength = src_maxlength
        self.tgt_maxlength = tgt_maxlength
        self.ans_maxlength = answer_maxlength

    def __getitem__(self, index):
        src_example = self.src[index]
        tgt_example = self.tgt[index]

        answer = src_example.split('[SEP]')[0].strip()
        passage = src_example.split('[SEP]')[1].strip()

        src_input_ids = self.tokenizer.encode(passage, add_special_tokens=True,
                                        max_length=self.src_maxlength, truncation=True,
                                       padding='max_length',return_tensors='pt')
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=True,
                                           max_length=self.ans_maxlength, truncation=True,
                                           padding='max_length', return_tensors='pt')
        tgt_input_ids = self.tokenizer.encode(tgt_example, add_special_tokens=True,
                                              max_length=self.tgt_maxlength, truncation=True,
                                              padding='max_length', return_tensors='pt')

        return src_input_ids, answer_ids, tgt_input_ids

    def __len__(self):
        return len(self.src)

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            src_tensor = torch.cat([feature[0] for feature in features])
            ans_tensor = torch.cat([feature[1] for feature in features])
            tgt_tensor = torch.cat([feature[2] for feature in features])
            return { "src_input_ids": src_tensor, "src_attention_mask": (src_tensor != 0).long(),
                     "answer_ids": ans_tensor, "answer_mask": (ans_tensor != 0).long(),
                     "tgt_input_ids": tgt_tensor, "tgt_attention_mask": (tgt_tensor != 0).long() }

        return fn


'''
s2s for AR model
'''
class S2S_AR_dataset(Dataset):
    def __init__(self, src, tgt, tokenizer, src_maxlength=144, tgt_maxlength=32):
        self.src = src
        self.tgt = tgt
        self.tokenizer = tokenizer
        self.src_maxlength = src_maxlength
        self.tgt_maxlength = tgt_maxlength

    def __getitem__(self, index):
        src_example = self.src[index]
        tgt_example = self.tgt[index]

        src_example.replace('<S_SEP>', '\n')
        tgt_example.replace('<S_SEP>', '\n')

        src_input_ids = self.tokenizer.encode(src_example, add_special_tokens=True,
                                        max_length=self.src_maxlength, truncation=True,
                                       padding='max_length', return_tensors='pt')
        tgt_input_ids = self.tokenizer.encode(tgt_example, add_special_tokens=True,
                                              max_length=self.tgt_maxlength, truncation=True,
                                              padding='max_length', return_tensors='pt')
        tgt_input_ids[tgt_input_ids == 1] = -100

        return src_input_ids, tgt_input_ids

    def __len__(self):
        return len(self.src)

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            src_tensor = torch.cat([feature[0] for feature in features])
            tgt_tensor = torch.cat([feature[1] for feature in features])
            # print("src shape:", src_tensor.shape)
            # print("tgt shape:", tgt_tensor.shape)
            return { "input_ids": src_tensor, "attention_mask": (src_tensor != 0).long(),
                     "labels": tgt_tensor}

        return fn

class S2S_dataset(Dataset):
    def __init__(self, src, tgt, tokenizer, src_maxlength=144, tgt_maxlength=32):
        self.src = src
        self.tgt = tgt
        self.tokenizer = tokenizer
        self.src_maxlength = src_maxlength
        self.tgt_maxlength = tgt_maxlength

    def __getitem__(self, index):
        src_example = self.src[index]
        tgt_example = self.tgt[index]

        src_input_ids = self.tokenizer.encode(src_example, add_special_tokens=True,
                                        max_length=self.src_maxlength, truncation=True,
                                       padding='max_length',return_tensors='pt')
        tgt_input_ids = self.tokenizer.encode(tgt_example, add_special_tokens=True,
                                              max_length=self.tgt_maxlength, truncation=True,
                                              padding='max_length', return_tensors='pt')

        return src_input_ids, tgt_input_ids

    def __len__(self):
        return len(self.src)

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            src_tensor = torch.cat([feature[0] for feature in features])
            tgt_tensor = torch.cat([feature[1] for feature in features])
            return { "src_input_ids": src_tensor, "src_attention_mask": (src_tensor != 0).long(),
                     "tgt_input_ids": tgt_tensor, "tgt_attention_mask": (tgt_tensor != 0).long() }

        return fn

class S2S_imp_dataset(Dataset):
    def __init__(self, src, tgt, ori_gen, tokenizer, src_maxlength=144, tgt_maxlength=32):
        self.src = src
        self.tgt = tgt
        self.ori_gen = ori_gen
        self.tokenizer = tokenizer
        self.src_maxlength = src_maxlength
        self.tgt_maxlength = tgt_maxlength

    def __getitem__(self, index):
        src_example = self.src[index]
        tgt_example = self.tgt[index]
        ori_gen_example = self.ori_gen[index]

        src_input_ids = self.tokenizer.encode(src_example, add_special_tokens=True,
                                        max_length=self.src_maxlength, truncation=True,
                                       padding='max_length',return_tensors='pt')
        tgt_input_ids = self.tokenizer.encode(tgt_example, add_special_tokens=True,
                                              max_length=self.tgt_maxlength, truncation=True,
                                              padding='max_length', return_tensors='pt')
        ori_gen_ids = self.tokenizer.encode(ori_gen_example, add_special_tokens=True,
                                            max_length=self.tgt_maxlength, truncation=True,
                                            padding='max_length', return_tensors='pt')

        return src_input_ids, tgt_input_ids, ori_gen_ids

    def __len__(self):
        return len(self.src)

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            src_tensor = torch.cat([feature[0] for feature in features])
            tgt_tensor = torch.cat([feature[1] for feature in features])
            ori_gen_tensor = torch.cat([feature[2] for feature in features])
            return { "src_input_ids": src_tensor, "src_attention_mask": (src_tensor != 0).long(),
                     "tgt_input_ids": tgt_tensor, "tgt_attention_mask": (tgt_tensor != 0).long(),
                     "ori_gen_ids": ori_gen_tensor}

        return fn

if __name__ == "__main__":
    pass