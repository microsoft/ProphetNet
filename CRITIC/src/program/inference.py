# Modified based on: https://github.com/wenhuchen/Program-of-Thoughts/blob/main/run_gsm8k.py
import json
import os
import pprint
from datetime import datetime
from time import sleep
from tqdm import tqdm
import argparse
from collections import Counter

from src.llms.api import llm
from src.datasets.dataset_loader import DatasetLoader
from src.utils import load_prompt, set_seed
from src.tools.interpreter_api import safe_execute
from src.program.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="gsm8k", type=str)
    parser.add_argument("--model", default="text-davinci-003", type=str)
    parser.add_argument("--prompt_type", default="pot", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    args = parser.parse_args()
    return args


def inference(args):
    # load prompt
    demo_prompt = load_prompt(args.data, args.prompt_type)

    # load dataset
    data_folder = f"data/{args.data}"
    os.makedirs(data_folder, exist_ok=True)
    data_file = f"data/{args.data}/{args.split}.json"

    if os.path.exists(data_file):
        print("Loading data from", data_file)
        dataset = DatasetLoader.load_dataset("json", data_files={args.split: data_file})[args.split]
    else:
        if args.data == "gsm8k":
            dataset = DatasetLoader.load_dataset(dataset_name=args.data, split=args.split, name="main")
        else:
            raise NotImplementedError(args.data)
        dataset.to_json(data_file)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        dataset = dataset.select(range(args.num_test_sample))
    print(dataset)

    # set start and end
    print('number of examples: ', len(dataset))

    # get filename
    now = datetime.now()
    dt_string = now.strftime("%m-%d_%H-%M")
    filename = f'outputs/{args.model}/{args.data}/{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}_s{args.start}_e{args.end}_{dt_string}.jsonl'
    os.makedirs(f'outputs/{args.model}/{args.data}', exist_ok=True)

    writer = open(filename, 'w')
    correct, wrong = 0, 0

    for idx, example in tqdm(enumerate(dataset)):
        if idx < args.start or (args.end != -1 and idx >= args.end):
            continue

        example = {**{'idx': idx}, **example}

        # construct prompt
        full_prompt = demo_prompt + f'Question: {example["question"]}' + '\n'
        if args.prompt_type == "pot":
            full_prompt += '# Python code, return answer' + '\n'
        elif args.prompt_type == "direct":
            full_prompt += 'Answer: '
        else:
            raise NotImplementedError(args.prompt_type)

        # call LLM
        result = llm(
            model=args.model,
            prompt=full_prompt,
            max_tokens=400,
            logprobs=1,
            temperature=args.temperature,
            stop=['\n\n'],
        )

        # parse result
        result = parse_api_result(result)
        prediction = None
        if args.prompt_type == "pot":
            ans, report = safe_execute(result[0])
            prediction = floatify_ans(ans)
        elif args.prompt_type == "direct":
            if result:
                prediction = normalize_answer(result[0].strip())
        else:
            raise NotImplementedError(args.prompt_type)

        # parse ground truth
        gt_cot, gt_ans = example['answer'].split("####") # GSM8k
        gt_cot, gt_ans = gt_cot.strip(), floatify_ans(gt_ans.strip())

        is_correct = finqa_equal(prediction, gt_ans)
        if is_correct:
            correct += 1
        else:
            wrong += 1

        sample = {'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans,
               'pred': prediction}
        if args.prompt_type == "pot":
            sample.update({'report': report, 'code': result})

        print("=" * 50)
        print(idx, is_correct)
        print(sample['question'])
        if args.prompt_type == "pot":
            print("```\n{}\n```".format(sample['code'][0]))
            print("Execution:", sample['report'])
        print("Output: answer =", sample['pred'])
        print()
        print("Gold CoT:", sample['gt_cot']) # print cot for reference
        print("Gold Ans:", sample['gt'])

        writer.write(json.dumps(sample) + '\n')
        writer.flush()

    writer.close()
    print()
    print(correct / (correct + wrong))


if __name__ == "__main__":
    args = parse_args()
    set_seed((args.seed))
    inference(args)