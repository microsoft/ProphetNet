import argparse

import numpy as np

from tqdm import tqdm
from multiprocessing import Pool
from eval_utils.mbr.pymteval import BLEUScore, NISTScore


def bleu_score(scorer, sent_sys, sents_ref):
    scorer.reset()
    scorer.append(sent_sys, [sents_ref])
    return scorer.score()


def cal_mbr(data):
    result = []
    for i in tqdm(range(len(data)), desc='process'):
        example_set = data[i]
        # print('example_set', example_set)
        score_dict = {}
        for idx in range(len(example_set)):
            y = example_set[idx]
            utility_lst = []
            for idx_x in range(len(example_set)):
                if idx_x != idx:
                    # BLEUScore(), BLEUScore(smoothing=1.0), NISTScore()
                    # NISTScore() spice is better but bleu4 is worse
                    utility_lst.append(bleu_score(BLEUScore(smoothing=1.0), example_set[idx_x], y))
            score_dict[idx] = np.array(utility_lst).mean()
        # print('score_dict', score_dict)
        best_y = sorted(score_dict.items(), key=lambda item: item[1])[-1]
        result.append(example_set[best_y[0]])
        # print('best_y', best_y)
        # print('result', result)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default="commongen", type=str)
    parser.add_argument('--num', default="4", type=int)
    parser.add_argument('--process', default="8", type=int)
    parser.add_argument('--exp_name', default="", type=str)
    args = parser.parse_args()

    raw_data = []
    for i in range(args.num):
        raw_data.append(open(f'source/data/{args.data_name}/gen_data/gen{i}.txt', 'r').readlines())

    data = []
    for i in range(len(raw_data[0])):
        temp = []
        for j in range(args.num):
            temp.append(raw_data[j][i].strip('\n'))
        data.append(temp)
    print(data[0])

    process_num = args.process
    pool = Pool(processes=process_num)
    results = []
    for i in range(process_num):
        start_idx = i * int(len(data) / process_num)
        end_idx = (i + 1) * int(len(data) / process_num)
        if i == process_num - 1:
            end_idx = len(data)
        input_data = data[start_idx:end_idx]
        results.append(pool.apply_async(cal_mbr, (input_data,)))
    pool.close()
    pool.join()
    print ("Sub-process(es) done.")

    sample_list = []
    for res in results:
        sample_list.extend(res.get())

    with open(f'source/data/{args.data_name}/gen_{args.exp_name}.txt', 'w', encoding='utf-8') as wp:
        for item in tqdm(sample_list, desc='writing'):
            wp.write(str(item) + '\n')
            wp.flush()
    wp.close()
