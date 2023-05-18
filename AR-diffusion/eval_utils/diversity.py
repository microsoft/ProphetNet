import string
import warnings
warnings.filterwarnings('ignore')


_tok_dict = {"(": "-LRB-", ")": "-RRB-",
             "[": "-LSB-", "]": "-RSB-",
             "{": "-LCB-", "}": "-RCB-"}
             

def _is_digit(w):
    for ch in w:
        if not(ch.isdigit() or ch == ','):
            return False
    return True


def fix_tokenization(text):
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok in _tok_dict.keys():
            output_tokens.append(_tok_dict[tok])
            i += 1
        elif tok == "\"":
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'"+input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(input_tokens) - 1 and _is_digit(input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ','+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.'+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[-1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[i + 1].isupper() and input_tokens[i + 2] == '.':
            # U . N . -> U.N.
            k = i+3
            while k+2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[i + 1][0] not in string.punctuation:
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)


def remove_duplicate(l_list, duplicate_rate):
    tk_list = [l.lower().split() for l in l_list]
    r_list = []
    history_set = set()
    for i, w_list in enumerate(tk_list):
        w_set = set(w_list)
        if len(w_set & history_set)/len(w_set) <= duplicate_rate:
            r_list.append(l_list[i])
        history_set |= w_set
    return r_list


def count_tokens(tokens):
    counter = {}
    for t in tokens:
        if t in counter.keys():
            counter[t] += 1
        else:
            counter[t] = 1
    return counter


def get_f1(text_a, text_b):
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if len(tokens_a) == 0 or len(tokens_b) == 0:
        return 1 if len(tokens_a) == len(tokens_b) else 0
    set_a = count_tokens(tokens_a)
    set_b = count_tokens(tokens_b)
    match = 0
    for token in set_a.keys():
        if token in set_b.keys():
            match += min(set_a[token], set_b[token])
    p = match / len(tokens_a)
    r = match / len(tokens_b)
    return 2.0 * p * r / (p + r + 1e-5)


from tqdm import tqdm
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='', help='data name')
    parser.add_argument('--num', type=int, default=4)

    args = parser.parse_args()
    return args

args = get_arguments()

if args.data == 'cnndm':
    path = '/wutong/DiffusionXY/data/cnn_dm/gen_data/'
    temp_gen = []
    for i in range(10):
        generated_list = []
        with open(path+f'gen{i}.txt', 'r') as fin:
            for line in tqdm(fin):
                buf = []
                for sentence in line.strip().split('< s _ sep >'):
                    sentence = fix_tokenization(sentence)
                    if any(get_f1(sentence, s) > 1.0 for s in buf):
                        continue
                    s_len = len(sentence.split())
                    if s_len <= 4:
                        continue
                    buf.append(sentence)
                if 0.7 < 1:
                    buf = remove_duplicate(buf, 0.7)
                if 0:
                    num_left = 0
                    trunc_list = []
                    for bit in buf:
                        tk_list = bit.split()
                        n = min(len(tk_list), num_left)
                        trunc_list.append(' '.join(tk_list[:n]))
                        num_left -= n
                        if num_left <= 0:
                            break
                else:
                    trunc_list = buf
                generated_list.append("\n".join(trunc_list))
        fin.close()
        temp_gen.append(generated_list)


else:
    path = f'/wutong/DiffusionXY/data/{args.data}/gen_data/'
    temp_gen = []
    for i in range(10):
        generated_list = []
        with open(path+f'gen{i}.txt', 'r') as fin:
            for line in tqdm(fin):
                generated_list.append(line.strip())
        fin.close()
        temp_gen.append(generated_list)


all_gen = []
for i in range(len(temp_gen[0])):
    temp = []
    for j in range(len(temp_gen)):
        temp.append(temp_gen[j][i].split())
    all_gen.append(temp)
print(all_gen[0])
print(len(all_gen), len(all_gen[0]))


import numpy as np

from fast_bleu import SelfBLEU
from multiprocessing import Pool


weights = {'four': (1/4., 1/4., 1/4., 1/4.)}
 
def func(gen_list):
    score_list = []
    for item in gen_list:
        score_list.append(SelfBLEU(item, weights).get_score()['four'])
    return score_list
 

if __name__ == "__main__":
    process_num = 50
    pool = Pool(processes=process_num)
    results = []
    for i in range(process_num):
        start_idx = i * int(len(all_gen) / process_num)
        end_idx = (i + 1) * int(len(all_gen) / process_num)
        if i == process_num - 1:
            end_idx = len(all_gen)
        input_data = all_gen[start_idx:end_idx]
        results.append(pool.apply_async(func, (input_data,)))
    pool.close()
    pool.join()
    print ("Sub-process(es) done.")

    score = []
    for res in results:     
        score.extend(res.get())

    print(np.mean(score, axis=0))
    print(np.mean(score))
