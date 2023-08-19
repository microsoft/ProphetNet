import os
import sys
import json
import openai
from datetime import datetime
import random
import time
import pprint
import numpy as np
import argparse

from src.datasets.dataset_loader import DatasetLoader
from src.llms.api import llm
from src.utils import set_seed, load_prompt
from src.qa.utils import em_f1_score, get_end_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="hotpot_qa", type=str)
    parser.add_argument("--model", default="text-davinci-003", type=str)
    parser.add_argument("--prompt_type", default="cot", type=str)
    parser.add_argument("--split", default="validation", type=str)
    parser.add_argument("--num_test_sample", default=500, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--num_sampling", default=1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    args = parser.parse_args()
    return args


def call_api(model, prompt, num_sampling, verbose=True, temperature=0):

    if temperature == 0:
        prediction = {"greedy": {}}
    else:
        prediction = {}
        prediction[f'temperature_{temperature}'] = {"text": [], "logprobs": [], "tokens": []}

    try:
        if temperature == 0: # greedy answer
            res = llm(prompt, model, stop=["\n\n"], logprobs=1)['choices'][0]
            prediction["greedy"]["text"] = res['text'].strip()
            assert prediction['greedy']['text'] != "", "Empty answer"
            # tokens & logprobs
            # end_idx = get_end_index(res['logprobs']['tokens'])
            # prediction["greedy"]["tokens"] = res['logprobs']['tokens'][:end_idx]
            # prediction["greedy"]["logprobs"] = res['logprobs']['token_logprobs'][:end_idx]

        else: # sampling
            res = llm(prompt, model, stop=["\n\n"], temperature=temperature, n=num_sampling, logprobs=1)
            for item in res['choices']:
                prediction[f"temperature_{temperature}"]["text"].append(item['text'].strip())
                # tokens & logprobs
                # end_idx = get_end_index(item['logprobs']['tokens'])
                # tokens = item['logprobs']['tokens'][:end_idx]
                # token_logprobs = item['logprobs']['token_logprobs'][:end_idx]
                # prediction[f"temperature_{temperature}"]["tokens"].append(tokens)
                # prediction[f"temperature_{temperature}"]["logprobs"].append(token_logprobs)
        return prediction
    except:
        return {}


def inference(args):

    # load prompt
    prompt = load_prompt(args.data, args.prompt_type)

    # load dataset
    data_folder = f"data/{args.data}"
    os.makedirs(data_folder, exist_ok=True)

    data_file = f"data/{args.data}/{args.split}.json"
    if os.path.exists(data_file):
        print("Loading data from", data_file)
        dataset = DatasetLoader.load_dataset("json", data_files={args.split: data_file})[args.split]
    else:
        # load data
        if data == "hotpot_qa":
            dataset = DatasetLoader.load_dataset(dataset_name=data, split=args.split, name="distractor")
        elif data == "trivia_qa": # BIG-Bench
            dataset = DatasetLoader.load_dataset(dataset_name=data, split=args.split, name="rc.nocontext")
        elif data in "ambig_qa":
            dataset = DatasetLoader.load_dataset(dataset_name=data, split=args.split) # question only, like BIG-Bench
        else:
            raise NotImplementedError(args.data)
        dataset.to_json(data_file)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        dataset = dataset.select(range(args.num_test_sample))
    print(dataset)

    # output file
    now = datetime.now()
    dt_string = now.strftime("%m-%d_%H-%M")
    save_folder = f"outputs/{args.model}/{args.data}"
    os.makedirs(save_folder, exist_ok=True)
    save_file = f"{save_folder}/{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}_s{args.start}_e{args.end}_{dt_string}.jsonl"

    # inference
    with open(save_file, "w", encoding="utf-8") as fp:
        for idx, sample in enumerate(dataset):
            if idx < args.start or (args.end != -1 and idx >= args.end):
                continue

            # remove keys
            entries_to_remove = ["context", "used_queries", "nq_doc_title"]
            for key in entries_to_remove:
                if key in sample:
                    sample.pop(key, None)

            # process question & answer
            if args.data == "ambig_qa":
                if sample['annotations']['type'][0] == "singleAnswer":
                    # single answer
                    answers = sample['nq_answer']
                    for ans in sample['annotations']['answer']:
                        answers.extend(ans)
                    sample['answer'] = list(set(answers))
                else:
                    # random choose a question with multiple answers
                    qa_pairs = sample['annotations']['qaPairs'][0]
                    rand_i = random.randint(0, len(qa_pairs['question'])-1)
                    sample['question'] = qa_pairs['question'][rand_i]
                    sample['answer'] = qa_pairs['answer'][rand_i]

            context = f"Q: {sample['question'].strip()}\nA: " 

            print(f"idx: {idx}")
            print(context, end="")

            prediction = call_api(args.model, prompt + context, num_sampling=args.num_sampling, temperature=args.temperature)

            sample['prediction'] = prediction

            if 'greedy' in prediction:
                print(prediction['greedy']['text'])
            print()

            fp.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    inference(args)