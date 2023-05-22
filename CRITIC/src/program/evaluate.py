import pprint
import json
import numpy as np
import random

from src.program.utils import finqa_equal, floatify_ans, normalize_answer
from src.tools.interpreter_api import safe_execute


def get_metrics(file_path, oracle=False):
    
    scores = []
    with open(file_path, 'r', encoding='utf-8') as fp:
        for idx, line in enumerate(fp):
            sample = json.loads(line)
            preds = []

            # for inference result evaluation
            if not isinstance(sample['pred'], list):
                sample['pred'] = [normalize_answer(sample['pred'])]

            # if None use previous answer
            for p in sample['pred']:
                preds.append(preds[-1] if (p is None and preds) else p)

            is_correct = [finqa_equal(p, sample['gt']) for p in preds]

            if oracle: # critic(oracle): only revise incorrect answer
                stop_idx = next((i for i, c in enumerate(is_correct) if c), None)
            else: # critic: stop if no correction twice (double check)
                stop_idx = next((i for i in range(2, len(preds)) if preds[i] == preds[i-1] == preds[i-2]), None)

            if stop_idx is not None:
                is_correct[stop_idx+1:] = [is_correct[stop_idx]] * (len(is_correct) - stop_idx - 1)

            scores.append(is_correct)
    
    print("num of samples:", len(scores))

    # output mean of each column of scores
    col_means= np.array(scores).mean(axis=0)
    print(list(np.round(col_means * 100, decimals=1)))
    print()


if __name__ == "__main__":
    ## text-davinci-003
    # file_path = "outputs/text-davinci-003/gsm8k/test_direct_greedy_-1_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/gsm8k/test_pot_-1_seed0.jsonl"
    file_path = "outputs/text-davinci-003/gsm8k/test_critic_-1_t0.5_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/gsm8k/test_critic_no-tools_-1_t0.5_seed0.jsonl" # critic no tools

    ## gpt-3.5-turbo
    # file_path = "outputs/gpt-3.5-turbo/gsm8k/test_direct_greedy_-1_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/gsm8k/test_pot_greedy_-1_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/gsm8k/test_critic-1_t0.5_seed0.jsonl" 
    # file_path = "outputs/gpt-3.5-turbo/gsm8k/test_critic_no_tools-1_t0.5_seed0.jsonl" 

    print(file_path)
    print("CRITIC:")
    get_metrics(file_path, oracle=False)
    print("CRITIC (oracle):")
    get_metrics(file_path, oracle=True)
