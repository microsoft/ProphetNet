import os
import argparse
from tqdm import tqdm
import string
import rouge

def get_arguments():
    parser = argparse.ArgumentParser()

    # out path
    parser.add_argument('--generate_path', type=str, default='', help='output path')
    parser.add_argument('--num_samples', type=int, default=50, help='sample query')

    # data args
    parser.add_argument('--data_path', type=str, default='', help='data path')
    parser.add_argument('--data_name', type=str, default='', help='data name')
    parser.add_argument('--batch_size', type=int, default=64, help='')

    # seed
    parser.add_argument('--seed', type=int, default=101, help='')
    parser.add_argument('--n_gpu', type=int, default=4, help='')

    args = parser.parse_args()
    return args

_tok_dict = {"(": "-lrb-", ")": "-rrb-",
             "[": "-lsb-", "]": "-rsb-",
             "{": "-lcb-", "}": "-rcb-",
             "[UNK]": "UNK", '&': '&amp;', '<': '&lt;', '>': '&gt;'}

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

def process_eval(args, gen_list, tgt_list):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2,
                            limit_length=False, apply_avg=True, weight_factor=1.2)

    max_score = []
    avg_score = []
    lowest_score = []
    best_gen_list = []
    # max score，avg score，lowest score
    for index, sentence in enumerate(tqdm(gen_list)):
        if index % args.num_samples == 0:
            max_score_dict = {'rouge_1':0.0,'rouge_2':0.0,'rouge_l':0.0}
            avg_score_dict = {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
            low_score_dict = {'rouge_1': 1.0, 'rouge_2': 1.0, 'rouge_l': 1.0}
        target = tgt_list[index // args.num_samples]
        scores = evaluator.get_scores([sentence], [[target]])
        rouge_1 = scores['rouge-1']['f']
        rouge_2 = scores['rouge-2']['f']
        rouge_l = scores['rouge-l']['f']
        # max
        if rouge_2 >= max_score_dict['rouge_2']:
            if rouge_2 != 0:
                max_score_dict['rouge_1'] = rouge_1
                max_score_dict['rouge_2'] = rouge_2
                max_score_dict['rouge_l'] = rouge_l
                best_sentence = sentence
            else:
                if rouge_1 >= max_score_dict['rouge_1']:
                    max_score_dict['rouge_1'] = rouge_1
                    max_score_dict['rouge_2'] = rouge_2
                    max_score_dict['rouge_l'] = rouge_l
                    best_sentence = sentence
        # avg
        avg_score_dict['rouge_1'] += rouge_1
        avg_score_dict['rouge_2'] += rouge_2
        avg_score_dict['rouge_l'] += rouge_l
        # min
        if rouge_2 < low_score_dict['rouge_2']:
            low_score_dict['rouge_1'] = rouge_1
            low_score_dict['rouge_2'] = rouge_2
            low_score_dict['rouge_l'] = rouge_l
        if (index + 1) % args.num_samples == 0:
            max_score.append(max_score_dict)
            best_gen_list.append(best_sentence)
            avg_score_dict['rouge_1'] = avg_score_dict['rouge_1'] / args.num_samples
            avg_score_dict['rouge_2'] = avg_score_dict['rouge_2'] / args.num_samples
            avg_score_dict['rouge_l'] = avg_score_dict['rouge_l'] / args.num_samples
            avg_score.append(avg_score_dict)
            lowest_score.append(low_score_dict)

    rouge_1 = 0
    rouge_2 = 0
    rouge_l = 0
    for score_dict in max_score:
        rouge_1 += score_dict['rouge_1']
        rouge_2 += score_dict['rouge_2']
        rouge_l += score_dict['rouge_l']
    max_rouge_1 = rouge_1 / len(max_score)
    max_rouge_2 = rouge_2 / len(max_score)
    max_rouge_l = rouge_l / len(max_score)

    rouge_1 = 0
    rouge_2 = 0
    rouge_l = 0
    for score_dict in avg_score:
        rouge_1 += score_dict['rouge_1']
        rouge_2 += score_dict['rouge_2']
        rouge_l += score_dict['rouge_l']
    avg_rouge_1 = rouge_1 / len(max_score)
    avg_rouge_2 = rouge_2 / len(max_score)
    avg_rouge_l = rouge_l / len(max_score)

    rouge_1 = 0
    rouge_2 = 0
    rouge_l = 0
    for score_dict in lowest_score:
        rouge_1 += score_dict['rouge_1']
        rouge_2 += score_dict['rouge_2']
        rouge_l += score_dict['rouge_l']
    min_rouge_1 = rouge_1 / len(max_score)
    min_rouge_2 = rouge_2 / len(max_score)
    min_rouge_l = rouge_l / len(max_score)

    scores = {'max_rouge_1':max_rouge_1, 'max_rouge_2':max_rouge_2, 'max_rouge_l':max_rouge_l,
              'min_rouge_1':min_rouge_1, 'min_rouge_2':min_rouge_2, 'min_rouge_l':min_rouge_l,
              'avg_rouge_1':avg_rouge_1, 'avg_rouge_2':avg_rouge_2, 'avg_rouge_l':avg_rouge_l }

    return scores, best_gen_list


def main():

    args = get_arguments()
    tgt = []
    test_tgt_path = os.path.join(args.data_path, args.data_name + "/org_data/test.tgt")
    with open(test_tgt_path, "r", encoding="utf-8") as ifile:
        for line in tqdm(ifile):
            line = line.strip()
            text = line
            # text = fix_tokenization(line)
            tgt.append(text)

    final_scores = {'max_rouge_1': 0.0, 'max_rouge_2': 0.0, 'max_rouge_l': 0.0,
              'min_rouge_1': 0.0, 'min_rouge_2': 0.0, 'min_rouge_l': 0.0,
              'avg_rouge_1': 0.0, 'avg_rouge_2': 0.0, 'avg_rouge_l': 0.0}

    tgt_offset = 0
    best_gen_list_total = []

    for i in range(args.n_gpu):

        gen_list_total = []
        for epoch in range(args.num_samples):
            gen_list = []
            gen_text_path = os.path.join(args.generate_path, "rank" + str(i)+"_gen_seed_" + str(args.seed) +
                                "_num" + str(args.num_samples) + "_epoch" + str(epoch+1) + ".txt")

            with open(gen_text_path, "r", encoding="utf-8") as ifile:
                for line in tqdm(ifile):
                    line = line.strip()
                    text = line
                    gen_list.append(text)
            gen_list_total.append(gen_list)

        gen_peace = []
        for index in range(len(gen_list_total[0])):
            for i in range(args.num_samples):
                gen_peace.append(gen_list_total[i][index])

        gen_len = len(gen_peace) // args.num_samples
        scores, best_gen_list = process_eval(args, gen_peace, tgt[tgt_offset:tgt_offset+gen_len])
        best_gen_list_total.extend(best_gen_list)
        for key, values in scores.items():
            final_scores[key] += values
        print("scores on gpu ", i)
        print(scores)
        tgt_offset += gen_len
    for key, values in final_scores.items():
        final_scores[key] = values / args.n_gpu

    print("final score :")
    print(final_scores)

    assert len(best_gen_list_total) == len(tgt)
    print("store best gen list ...")
    out_path = os.path.join(args.generate_path, "best_gen_list.txt")
    with open(out_path, 'w') as f:
        for sentence in best_gen_list_total:
            f.write(sentence + '\n')



if __name__ == "__main__":
    main()