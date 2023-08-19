import json
import os
import pprint
import random
from datetime import datetime
from time import sleep
from tqdm import tqdm
import argparse
from collections import Counter

from src.llms.api import llm
from src.datasets.dataset_loader import DatasetLoader
from src.utils import set_seed, load_jsonl
from src.toxicity.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="toxicity", type=str)
    parser.add_argument("--model", default="text-davinci-003", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=1000, type=int, help="-1 for full data")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_tokens", default=20, type=int, help="max number of continuation tokens")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0.9, type=float)
    parser.add_argument("--num_sampling", default=25)
    args = parser.parse_args()
    return args


def inference(args):
    # load dataset
    data_folder = f"data/{args.data}"
    os.makedirs(data_folder, exist_ok=True)
    data_file = f"data/{args.data}/{args.split}.jsonl"

    dataset = []
    for sample in load_jsonl(data_file):
        dataset.append(sample)

    # randomly select samples from dataset
    if args.num_test_sample > 0:
        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        dataset = [dataset[idx] for idx in idxs[:args.num_test_sample]]
    print('number of samples: ', len(dataset))

    # get filename
    now = datetime.now()
    dt_string = now.strftime("%m-%d_%H-%M")
    filename = f'outputs/{args.model}/{args.data}/{args.split}_{args.num_test_sample}_seed{args.seed}_s{args.start}_e{args.end}_{dt_string}.jsonl'
    os.makedirs(f'outputs/{args.model}/{args.data}', exist_ok=True)

    writer = open(filename, 'w')

    for idx, sample in tqdm(enumerate(dataset)):
        if idx < args.start or (args.end != -1 and idx >= args.end):
            continue
        
        prompt = sample['prompt']['text'].strip()
        print(prompt)

        # get result
        result = llm(
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            n=args.num_sampling,
            logprobs=1,
            stop=['\n'],
        )

        if not result: # skip failed request
            continue

        prediction = result['choices']

        print(">>", prediction[0]['text'])
        print()

        # remove keys in log_probs: top_logprobs, tokens, text_offset
        for choice in prediction:
            choice['logprobs'] = {k: choice['logprobs'][k] for k in choice['logprobs'] if k in ['token_logprobs']}

        sample['prediction'] = prediction 
        writer.write(json.dumps(sample) + '\n')
        writer.flush()

    writer.close()


if __name__ == "__main__":
    args = parse_args()
    set_seed((args.seed))
    inference(args)