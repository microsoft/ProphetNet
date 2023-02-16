
from pytorch_transformers import BertTokenizer
import os
import numpy as np
import sys
from tqdm import tqdm
import emoji
import re
from sklearn.model_selection import train_test_split
# from nltk.tokenize.treebank import TreebankWordDetokenizer
# from nltk.tokenize import TweetTokenizer


def infer_prefix(mode: str = 'finetune') -> str:
    assert mode in ['finetune', 'pretrain'], 'mode must be either `finetune` or `pretrain`'
    if mode == 'finetune':
        if sys.platform.startswith('win'):
            prefix = 'D:/dialogue/finetune'
        else:
            prefix = '/home/v-wchen2/Data/dialogue/finetune'
    else:
        if sys.platform.startswith('win'):
            prefix = 'D:/dialogue/pretrain'
        else:
            prefix = '/home/v-wchen2/Data/dialogue/pretrain'
    return prefix


# input and output length
def dialog_input_output_len(fin: str, has_knowledge: bool = False) -> None:
    fin = open(fin, 'r', encoding='utf-8').readlines()
    src_len, tgt_len = [], []
    if has_knowledge:
        for line in fin:
            src1, src2, tgt = line.strip('\n').split('\t')
            src_len.append(len(src1.split()) + len(src2.split()))
            tgt_len.append(len(tgt.split()))
    else:
        for line in fin:
            src, tgt = line.strip('\n').split('\t')
            src_len.append(len(src.split()))
            tgt_len.append(len(tgt.split()))
    src_len, tgt_len, quantile = np.array(src_len), np.array(tgt_len), .99
    print('{}% quantile src/tgt length: {}/{}, max src/tgt length: {}/{}'.format(
        int(quantile * 100), int(np.quantile(src_len, 0.99)), int(np.quantile(tgt_len, 0.99)),
        int(np.max(src_len)), int(np.max(tgt_len))))


# input and output length for twitter
def dialog_input_output_len_twitter(fin: str):
    fin = open(fin, 'r', encoding='utf-8').readlines()
    src_len = np.array([len(line.strip().split()) for i, line in enumerate(fin) if i % 2 == 0])
    tgt_len = np.array([len(line.strip().split()) for i, line in enumerate(fin) if i % 2 == 1])
    quantile = .99
    print('{}% quantile src/tgt length: {}/{}, max src/tgt length: {}/{}'.format(
        int(quantile * 100), int(np.quantile(src_len, 0.99)), int(np.quantile(tgt_len, 0.99)),
        int(np.max(src_len)), int(np.max(tgt_len))))


# input & output & tokenizer loader
def prepare(fin: str, src_fout: str, tgt_fout: str) -> tuple:
    fin = open(fin, 'r', encoding='utf-8').readlines()
    src_fout = open(src_fout, 'w', encoding='utf-8')
    tgt_fout = open(tgt_fout, 'w', encoding='utf-8')
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    return fin, src_fout, tgt_fout, tok


def split_line_base(string: str, tok) -> list:
    return [' '.join(_) for _ in [tok.tokenize(_.strip()) for _ in string.split('__eou__')]]


def split_line(line: str, tok: BertTokenizer, sep: str, sep_all: bool, has_knowledge: bool = False) -> tuple:
    if has_knowledge:
        src1, src2, tgt = line.strip('\n').split('\t')
        if sep_all:
            src_line = sep.join(split_line_base(src1, tok) + split_line_base(src2, tok))
        else:
            src_line = ' '.join(split_line_base(src1, tok)) + sep + ' '.join(split_line_base(src2, tok))
    else:
        src, tgt = line.strip('\n').split('\t')
        if sep_all:
            src_line = sep.join(split_line_base(src, tok))
        else:
            src_line = ' '.join(split_line_base(src, tok))
    tgt_line = sep.join(split_line_base(tgt, tok))
    return src_line, tgt_line


def detokenize_bert(tokens: list) -> str:
    return ' '.join([x for x in tokens]).replace(' ##', '')


def convert_daily_dialog(
        fin: str,
        src_fout: str,
        tgt_fout: str,
        sep: str = ' [SEP] ',
        sep_all: bool = True,
        test: bool = False,
        has_knowledge: bool = False,
        max_src: int = 512,
        max_tgt: int = 128) -> None:
    fin, src_fout, tgt_fout, tok = prepare(fin, src_fout, tgt_fout)
    for line in tqdm(fin):
        src_line, tgt_line = split_line(line, tok, sep, sep_all, has_knowledge)
        if test:
            src_line = ' '.join(src_line.split()[: max_src - 1])
            tgt_line = ' '.join(tgt_line.split()[: max_tgt - 1]).replace(' ##', '')
        src_fout.write('{}\n'.format(src_line))
        tgt_fout.write('{}\n'.format(tgt_line))
    src_fout.close()
    tgt_fout.close()


def convert_persona_chat(fin, src_fout, tgt_fout, sep=' [SEP] ', sep_all=True, test=False, has_knowledge=True):
    convert_daily_dialog(fin, src_fout, tgt_fout, sep, test, sep_all, has_knowledge)


def convert_dstc7_avsd(
        fin: str,
        src_fout: str,
        tgt_fout: str,
        multi_ref_tgt_fout=None,
        sep: str = ' [SEP] ',
        sep_all: bool = True,
        test: bool = False,
        has_knowledge: bool = True,
        max_src: int = 512) -> None:
    if not test:
        convert_daily_dialog(fin, src_fout, tgt_fout, sep, sep_all, test, has_knowledge)
    else:
        multi_ref_tgt_fout = open(multi_ref_tgt_fout, 'w', encoding='utf-8')
        fin, src_fout, tgt_fout, tok = prepare(fin, src_fout, tgt_fout)
        for line in tqdm(fin):
            src_line, tgt_line = split_line(line, tok, sep, sep_all, has_knowledge)
            src_line = ' '.join(src_line.split()[: max_src])
            tgt_line = [instance.strip() for instance in tgt_line.replace(' ##', '').split('|')]
            src_fout.write('{}\n'.format(src_line))
            tgt_fout.write('{}\n'.format(tgt_line[0]))
            multi_ref_tgt_fout.write('{}\n\n'.format('\n'.join(tgt_line)))
        multi_ref_tgt_fout.close()


def check(processed_path: str) -> None:
    with open(os.path.join(processed_path, 'test.src'), encoding='utf-8') as f:
        _src = f.readlines()
    with open(os.path.join(processed_path, 'test.tgt'), encoding='utf-8') as f:
        _tgt = f.readlines()
    _max_src, _max_tgt = -1, -1
    _max_src_tgt, _max_tgt_src = -1, -1
    for __src, __tgt in zip(_src, _tgt):
        __src_len = len(__src.strip('\n').split())
        __tgt_len = len(__tgt.strip('\n').split())
        if __src_len > _max_src:
            _max_src = __src_len
            _max_src_tgt = __tgt_len
        if __tgt_len > _max_tgt:
            _max_tgt = __tgt_len
            _max_tgt_src = __src_len
    print('{}\nboundary shape src: ({}, {})\nboundary shape tgt: ({}, {})'.format(
        processed_path, _max_src, _max_src_tgt, _max_tgt_src, _max_tgt))


def write_str_list(array: list, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in tqdm(array):
            f.write(text + '\n')


def make_dir(path):
    if isinstance(path, str):
        if not os.path.exists(path):
            os.makedirs(path)
    elif isinstance(path, list):
        for p in path:
            make_dir(p)


# convert reddit data
def convert_reddit_for_finetune(fin: str, fout: str) -> None:
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    fin = open(fin, 'r', encoding='utf-8')
    # if directory not exits, make one
    make_dir(fout)
    train_src = open(os.path.join(fout, 'train.src'), 'w', encoding='utf-8')
    train_tgt = open(os.path.join(fout, 'train.tgt'), 'w', encoding='utf-8')
    valid_src = open(os.path.join(fout, 'valid.src'), 'w', encoding='utf-8')
    valid_tgt = open(os.path.join(fout, 'valid.tgt'), 'w', encoding='utf-8')
    for line in tqdm(fin, total=146832759):
        _, contexts, response = line.strip().split('\t')
        contexts = [' '.join(
            tok.tokenize(' '.join(context.strip().split()[1:]))) for context in contexts.split('EOS')]
        response = ' '.join(tok.tokenize(' '.join(response.split(' ')[1:])))
        if np.random.rand() < 0.05:
            valid_src.write(' [SEP] '.join(contexts) + '\n')
            valid_tgt.write(response + '\n')
        else:
            train_src.write(' [SEP] '.join(contexts) + '\n')
            train_tgt.write(response + '\n')


def convert_reddit_for_pretrain(fin: str, fout: str) -> None:
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    fin = open(fin, 'r', encoding='utf-8')
    # if directory not exits, make one
    make_dir(fout)
    train_src = open(os.path.join(fout, 'train.src'), 'w', encoding='utf-8')
    valid_src = open(os.path.join(fout, 'valid.src'), 'w', encoding='utf-8')
    for line in tqdm(fin, total=146832759):
        _, contexts, response = line.strip().split('\t')
        contexts = [' '.join(
            tok.tokenize(' '.join(context.strip().split()[1:]))) for context in contexts.split('EOS')]
        response = ' '.join(tok.tokenize(' '.join(response.split(' ')[1:])))
        dialog = contexts + [response]
        if np.random.rand() < 0.05:
            valid_src.write(' [SEP] '.join(dialog) + '\n')
        else:
            train_src.write(' [SEP] '.join(dialog) + '\n')


# convert twitter for fine-tuning
def convert_twitter_for_finetune(fin: str) -> None:
    with open(fin, 'r', encoding='utf-8') as f:
        fin = f.readlines()
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    src_lines, tgt_lines, num_of_lines, min_len = [], [], int(len(fin) / 2), 3
    # extra space pattern
    space_pattern = re.compile(r':\S+?:')
    for num_of_line in tqdm(range(num_of_lines)):
        # https://github.com/marsan-ma/chat_corpus
        # This is a chat corpus collection from various open sources
        # all files are composed of question-answer pairs, where odd lines are questions, even lines are answers.
        src, tgt = fin[2 * num_of_line].strip(), fin[2 * num_of_line + 1].strip()
        # remove emoji
        src, tgt = space_pattern.sub(' ', emoji.demojize(src)), space_pattern.sub(' ', emoji.demojize(tgt))
        src_line, tgt_line = tok.tokenize(src), tok.tokenize(tgt)
        if len(src_line) < min_len and len(tgt_line) < min_len:
            continue
        src_lines.append(' '.join(src_line))
        tgt_lines.append(' '.join(tgt_line))
    train_src, valid_src, train_tgt, valid_tgt = train_test_split(
        src_lines, tgt_lines, test_size=0.1, random_state=100)
    if not os.path.exists(os.path.join(PRETRAIN_PREFIX_PATH, 'twitter/processed')):
        os.makedirs(os.path.join(PRETRAIN_PREFIX_PATH, 'twitter/processed'))
    write_str_list(train_src, os.path.join(PRETRAIN_PREFIX_PATH, 'twitter/processed/train.src'))
    write_str_list(valid_src, os.path.join(PRETRAIN_PREFIX_PATH, 'twitter/processed/valid.src'))
    write_str_list(train_tgt, os.path.join(PRETRAIN_PREFIX_PATH, 'twitter/processed/train.tgt'))
    write_str_list(valid_tgt, os.path.join(PRETRAIN_PREFIX_PATH, 'twitter/processed/valid.tgt'))


FINETUNE_PREFIX_PATH = infer_prefix(mode='finetune')
PRETRAIN_PREFIX_PATH = infer_prefix(mode='pretrain')
