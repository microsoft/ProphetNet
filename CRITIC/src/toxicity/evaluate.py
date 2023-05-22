# modified based on: https://github.com/GXimingLu/Quark/blob/main/sample.py
import json
import numpy as np
from tqdm import tqdm

from src.utils import load_jsonl
from src.toxicity.utils import *


def evaluate(file_path, output=False, max_iter=4):

    total_iter = max_iter + 1 # init iter + critic iter
    all_samples = {i: [] for i in range(total_iter)}

    for sample in tqdm(load_jsonl(file_path)):

        # get maximum toxicity and toxicity probability of iteration i
        cur_sample = {i: [] for i in range(total_iter)}

        for pred in sample['prediction']:
            # skip error
            if 'info' not in pred[-1] and len(pred) < total_iter:
                continue

            if 'info' in pred[-1]:
                pred = pred[:-1]

            # for sanity
            if all('toxicity' in pred[itr] and len(pred[itr]['text'].strip().split(" ")) > 2 for itr in range(len(pred))):
                for cur_iter in range(total_iter):
                    cur_sample[cur_iter].append(pred[cur_iter] if cur_iter < len(pred) else pred[-1])

        # add to all_samples
        for i in range(total_iter):
            if len(cur_sample[i]) > 0:
                all_samples[i].append({"prompt": sample["prompt"]['text'], "continuation": cur_sample[i]})
            
    # print metrics 
    all_metrics = []
    for itr, samples in all_samples.items():
        print(f"{'-' * 20} Iter {itr} {'-' * 20}")
        print("num samples:", len(samples))
        metrics = {
            'avg_max_tox': round(toxicity(samples)[0], 3),
            'avg_tox_prob': round(toxicity(samples)[1], 3),
            'ppl': round(perplexity(samples), 2),
            'dist2': round(distinctness(samples)[1], 2),
            'dist3': round(distinctness(samples)[2], 2)
        }
        all_metrics.append(metrics)
        print(f"{metrics['avg_max_tox']:.3f} & {metrics['avg_tox_prob']:.3f} & {metrics['ppl']:.2f} & {metrics['dist2']:.2f} & {metrics['dist3']:.2f}\n")

    print("\n", "-" * 20, "All Iter", "-" * 20)
    for metric_name in ['avg_max_tox', 'avg_tox_prob', 'ppl', 'dist2', 'dist3']:
        print(f"'{metric_name.capitalize()}': {[metric[metric_name] for metric in all_metrics]},")

    # save all_samples
    if output:
        # Extract continuation text from all_samples
        for i in range(total_iter):
            for j in range(len(all_samples[i])):
                all_samples[i][j]['continuation'] = [x['text'] for x in all_samples[i][j]['continuation']]
        
        # Save all_samples to a JSON file
        out_file = file_path.replace(".jsonl", "_all_samples.json")
        with open(out_file, "w") as f:
            json.dump(all_samples, f, indent=2)


if __name__ == "__main__":

    ## gpt-3.5-turbo
    # file_path = "outputs/gpt-3.5-turbo/toxicity/test_critic_v1_1000_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/toxicity/test_critic_no-tool_1000_seed0.jsonl" # no tools

    ## text-davinci-003
    file_path = "outputs/text-davinci-003/toxicity/test_critic_1000_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/toxicity/test_critic_no-tool_1000_seed0.jsonl" # no tools

    evaluate(file_path, output=True)
 