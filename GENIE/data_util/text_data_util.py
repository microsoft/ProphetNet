import os
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from util import logger
from train_util import dist_util
import numpy as np

def load_data_text(data_args, emb_model=None, padding_mode='max_len', split='train', tokenizer=None, return_hidden=False):

    '''
    init embedding model
    '''
    if split == 'train':
        if data_args.token_emb_type == 'random' and emb_model is None:
            print('loading initialized random embeddings. ')
            emb_model = None
        elif data_args.token_emb_type == 'pretrain' and emb_model is not None:
            print('loading embeddings from pretraining embedding. ')

    '''
    load query text dataset (only query) according to split
    '''
    if data_args.model_arch == 'transformer':
        dataset = get_query_corpus(data_args, padding_mode=padding_mode, split=split,
                                               tokenizer=tokenizer)
    elif data_args.model_arch == 's2s_CAT':
        dataset = get_pandq_corpus(data_args, padding_mode=padding_mode, split=split,
                                         tokenizer=tokenizer)
    else:
        return NotImplementedError
    '''
    load embedding model
    '''
    # if data_args.token_emb_type == 'random' and emb_model is None:
    #     emb_model = torch.nn.Embedding(tokenizer.vocab_size, data_args.in_channel)
    #     print('initializing the random embeddings', emb_model)
    #     torch.nn.init.normal_(emb_model.weight)
    #     path_save = f'{data_args.checkpoint_path}/random_emb.torch'
    #     print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
    #     torch.save(emb_model.state_dict(), path_save)
    # elif data_args.token_emb_type == 'pretrain':
    #     # TODO: finish pretrain embedding setting
    #     print('initializing the pretrain embeddings', emb_model)
    # else:
    #     return NotImplementedError

    if return_hidden:
        '''
        encode input ids to input embedding
        load temp dataloader to input in embedding model, getting hidden_state (word embedding)
        '''
        query_dataloader = DataLoader(dataset, batch_size=1024, drop_last=False,
                                num_workers=20, collate_fn=Question_dataset.get_collate_fn())
        emb_model.to(dist_util.dev())
        hidden_state_set = []
        with torch.no_grad():
            for k, (ids, text_ids, text_mask) in enumerate(tqdm(query_dataloader)):
                hidden_state = emb_model(text_ids.long().to(dist_util.dev()))
                # hidden_state :(batch_size, seq_len, hidden_size)
                hidden_state = hidden_state.detach().cpu()
                hidden_state_set.append(hidden_state)
        hidden_state_set = torch.cat(hidden_state_set, dim=0)

        '''
        load query embedding dataloader
        '''
        query_emb_dataset = Text_Hidden_dataset(dataset, hidden_state_set)
        # query_dataloader = DataLoader(query_emb_dataset, batch_size=data_args.batch_size, drop_last=False,
        #                               num_workers=20, collate_fn=Text_Hidden_dataset.get_collate_fn())

        emb_model.cpu()
        return query_emb_dataset, emb_model

    else:
        return dataset, emb_model

    # for input_ids in dataset['word_ids']:
    #     if data_args.experiment.startswith('random'):
    #         hidden_state = model(torch.tensor(input_ids))
    #     elif data_args.experiment == 'gpt2_pre_compress':
    #         input_ids2 = torch.tensor(input_ids).to(model.device)
    #         input_embs = model.transformer.wte(input_ids2)  # input_embs
    #         hidden_state = model.down_proj(input_embs)
    #         hidden_state = hidden_state * data_args.emb_scale_factor
    #     elif data_args.experiment == 'glove':
    #         hidden_state = model(torch.tensor(input_ids))
    #     result_train_lst.append({'input_ids': input_ids, 'hidden_states': hidden_state.cpu().tolist()})


def get_query_corpus(args, padding_mode, split, tokenizer):

    questions = []
    if split == 'train':
        print("***** load train query dataset*****")
        train_question_path = os.path.join(args.data_path, "train.query.txt")
        with open(train_question_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                id, text = line.split('\t')
                questions.append([int(id), text])
    elif split == 'dev':
        dev_question_path = os.path.join(args.data_path, "dev.query.txt")
        with open(dev_question_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                id, text = line.split('\t')
                questions.append([int(id), text])
    else:
        print("no such split of data...")
        exit(0)

    print("example of questions text: ", questions[50])

    if padding_mode == 'max_len':
        dataset = Question_dataset(questions, tokenizer, maxlength=args.text_max_len)
    elif padding_mode == 'block':
        print("padding block is under realization")
        pass
    else:
        return NotImplementedError

    print("example of questions id lists: ", dataset[50])
    print("total query dataset len :", len(dataset))

    return dataset

def get_pandq_corpus(args, padding_mode, split, tokenizer):
    questions = []
    if split == 'train':
        print("***** load train query dataset*****")
        train_question_path = os.path.join(args.data_path, "train.query.txt")
        with open(train_question_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                id, text = line.split('\t')
                questions.append([int(id), text])
        print("***** load query to passage qrel data*****")
        qrel_path = os.path.join(args.data_path, "qrels.train.tsv")
        qids_to_relevant_passageids = load_train_reference_from_stream(qrel_path)


    elif split == 'dev':
        dev_question_path = os.path.join(args.data_path, "dev.query.txt")
        with open(dev_question_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                id, text = line.split('\t')
                questions.append([int(id), text])
        print("***** load query to passage qrel data*****")
        qrel_path = os.path.join(args.data_path, "qrels.dev.tsv")
        qids_to_relevant_passageids = load_dev_reference_from_stream(qrel_path)
    else:
        print("no such split of data...")
        exit(0)

    print("***** load passages dataset*****")
    passage_title_path = os.path.join(args.data_path, "para.title.txt")
    passage_ctx_path = os.path.join(args.data_path, "para.txt")
    passage_title = load_id_text(passage_title_path)
    passages = {}
    with open(passage_ctx_path) as inp:
        for line in tqdm(inp):
            line = line.strip()
            id, text = line.split('\t')
            passages[int(id)] = (text, passage_title.get(id, '-'))

    print("example of questions text: ", questions[50])
    qid = questions[50][0]
    pid = qids_to_relevant_passageids[qid][0]
    print("example of passage text  --title: ", passages[pid][1])
    print("example of passage text  --passage: ", passages[pid][0])


    if padding_mode == 'max_len':
        dataset = PandQ_dataset(questions, qids_to_relevant_passageids, passages,
                                tokenizer, q_maxlength=args.text_max_len, p_maxlength=args.pas_max_len)
    elif padding_mode == 'block':
        print("padding block is under realization")
        pass
    else:
        return NotImplementedError

    print("example of questions id lists: ", dataset[50][0])
    print("example of passage id lists: ", dataset[50][1])
    print("total query dataset len :", len(dataset))

    return dataset


class Text_Hidden_dataset(Dataset):
    def __init__(self, query_dataset, hidden_state_set):
        self.query_dataset = query_dataset
        self.hidden_state_set = hidden_state_set

    def __getitem__(self, index):
        example = self.query_dataset[index]
        hidden_state = self.hidden_state_set[index]

        return example[0], example[1], hidden_state

    '''
    query_ids : len(np list) = batch_size query
    input_ids : (batch_size, seq_len) 
    attention_mask : (batch_size, seq_len) 
    hidden_state : (batch_size, seq_len, embedding) 
    '''
    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            id_list = [feature[0] for feature in features]
            q_tensor = torch.cat([feature[1] for feature in features])
            hidden_state = torch.cat([feature[2] for feature in features])
            return {"query_ids":np.array(id_list), "input_ids":q_tensor,
                    "attention_mask":(q_tensor != 0).long(), "hidden_state":hidden_state}

        return fn

class PandQ_dataset(Dataset):
    def __init__(self, questions, qids_to_relevant_passageids, passages,
                 tokenizer, q_maxlength=32, p_maxlength=144):
        self.questions = questions
        self.tokenizer = tokenizer
        self.q_maxlength = q_maxlength
        self.p_maxlength = p_maxlength
        self.qids_to_relevant_passageids = qids_to_relevant_passageids
        self.passages = passages

    def __getitem__(self, index):
        q_example = self.questions[index]
        query_id = q_example[0]
        q_input_ids = self.tokenizer.encode(q_example[1], add_special_tokens=True,
                                        max_length=self.q_maxlength, truncation=True,
                                       padding='max_length',return_tensors='pt')
        rel_passage_id = self.qids_to_relevant_passageids[query_id][0]
        passage_example = self.passages[rel_passage_id]
        text = passage_example[0]
        title = passage_example[1]
        p_input_ids = self.tokenizer.encode(title, text_pair=text, add_special_tokens=True,
                                          max_length=self.p_maxlength, truncation=True,
                                          padding='max_length', return_tensors='pt')

        return q_input_ids, p_input_ids

    def __len__(self):
        return len(self.questions)

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            q_tensor = torch.cat([feature[0] for feature in features])
            p_tensor = torch.cat([feature[1] for feature in features])
            return { "q_input_ids": q_tensor, "q_attention_mask": (q_tensor != 0).long(),
                     "p_input_ids": p_tensor, "p_attention_mask": (p_tensor != 0).long() }

        return fn



class Question_dataset(Dataset):
    def __init__(self, questions, tokenizer,maxlength=32):
        self.questions = questions
        self.tokenizer = tokenizer
        self.maxlength = maxlength
    def __getitem__(self, index):
        example = self.questions[index]
        input_ids = self.tokenizer.encode(example[1], add_special_tokens=True,
                                        max_length=self.maxlength, truncation=True,
                                       padding='max_length',return_tensors='pt')
        return example[0], input_ids

    def __len__(self):
        return len(self.questions)

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            id_list = [feature[0] for feature in features]
            q_tensor = torch.cat([feature[1] for feature in features])
            return np.array(id_list), q_tensor, (q_tensor != 0).long()
        return fn


def load_dev_reference_from_stream(path_to_reference):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    qids_to_relevant_passageids = {}
    with open(path_to_reference, 'r') as f:
        for l in f:
            try:
                l = l.strip().split('\t')
                qid = int(l[0])
                if qid in qids_to_relevant_passageids:
                    pass
                else:
                    qids_to_relevant_passageids[qid] = []
                qids_to_relevant_passageids[qid].append(int(l[1]))
            except:
                raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids


def csv_reader(fd, delimiter='\t', trainer_id=0, trainer_num=1):
    def gen():
        for i, line in tqdm(enumerate(fd)):
            if i % trainer_num == trainer_id:
                slots = line.rstrip('\n').split(delimiter)
                if len(slots) == 1:
                    yield slots,
                else:
                    yield slots
    return gen()
def load_train_reference_from_stream(input_file, trainer_id=0, trainer_num=1):
    """Reads a tab separated value file."""
    with open(input_file, 'r', encoding='utf8') as f:
        reader = csv_reader(f, trainer_id=trainer_id, trainer_num=trainer_num)
        #headers = 'query_id\tpos_id\tneg_id'.split('\t')

        #Example = namedtuple('Example', headers)
        qrel = {}
        for [topicid, _, docid, rel] in reader:
            topicid = int(topicid)
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(int(docid))
            else:
                qrel[topicid] = [int(docid)]
    return qrel

def load_id_text(file_name):
    """load tsv files"""
    id_text = {}
    with open(file_name) as inp:
        for line in tqdm(inp):
            line = line.strip()
            id, text = line.split('\t')
            id_text[id] = text
    return id_text

