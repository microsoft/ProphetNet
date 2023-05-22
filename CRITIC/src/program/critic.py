import json
import os
import pprint
from datetime import datetime
from time import sleep
from tqdm import tqdm
import argparse
from collections import Counter
from distutils.util import strtobool

from src.llms.api import llm
from src.datasets.dataset_loader import DatasetLoader
from src.utils import load_prompt, set_seed, load_jsonl
from src.tools.interpreter_api import safe_execute
from src.program.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="gsm8k", type=str)
    parser.add_argument("--model", default="text-davinci-003", type=str)
    parser.add_argument("--prompt_type", default="pot", type=str)
    parser.add_argument("--critic_type", default="critic", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--max_iter", default=4, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--use_tool", type=strtobool, default=True)
    args = parser.parse_args()
    return args


def critic(args):
    # load prompt
    prompt = load_prompt(args.data, args.critic_type)

    print(prompt)
    print("%" * 30, "Critic", "%" * 30)

    # input and output file
    now = datetime.now()
    dt_string = now.strftime("%m-%d_%H-%M")
    init_file = f'outputs/{args.model}/{args.data}/{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}.jsonl'
    out_file = f'outputs/{args.model}/{args.data}/{args.split}_{args.critic_type}_{args.num_test_sample}_t{args.temperature}_seed{args.seed}_s{args.start}_e{args.end}_{dt_string}.jsonl'

    writer = open(out_file, 'w')

    for idx, sample in enumerate(load_jsonl(init_file)):
        if idx < args.start or (args.end != -1 and idx >= args.end):
            continue
        print("\n\n" + "=" * 30, "Idx", idx, "=" * 30)

        # add idx to the begining of sample
        sample = {**{'idx': idx}, **sample}

        for itr in range(1, args.max_iter + 1):
            if itr == 1:
                print("Is initial program correct:", sample['gt'] == sample['pred'])
                sample['pred'] = [sample['pred']]
                sample['report'] = [sample['report']]
            print("\n" + "-" * 20, "iteration", itr, "-" * 20)
            
            # criticize latest answer that is not "None"
            base_idx = itr - 1
            while base_idx > 0 and sample['pred'][base_idx] is None:
                base_idx -= 1
            print("Correct based on iter:", base_idx)

            previous_code = remove_comment(sample['code'][base_idx])

            # construct prompt
            context = f"Question: {sample['question']}\n"
            context += f"```python\n{previous_code}\n```\n"
            if args.use_tool:
                context += f"Execution: {sample['report'][base_idx]}\n"
                context += f"Output: answer = {floatify_ans(sample['pred'][base_idx])}\n"
            context += "\nWhat's the problem with the above code?\n\n"
            prompt_critic = prompt + context
            print(context, end="")

            # verify previous code
            result = llm(
                model=args.model,
                prompt=prompt_critic,
                max_tokens=500,
                logprobs=1,
                temperature=args.temperature,
                n=1,
                stop=["Here's", "---"],
            )
            context = parse_api_result(result)[0] if result else ""

            # if context not end with a "\n", add "\n"
            if context and context[-1] != "\n":
                context += "\n"

            # generate new code
            context += "Here's a better solution:\n```python\n"
            prompt_critic += context
            print(context, end="")

            result = llm(
                model=args.model,
                prompt=prompt_critic,
                max_tokens=400,
                logprobs=1,
                temperature=args.temperature,
                n=1,
                stop=["```", "---"]
            )

            # excute new code
            code = parse_api_result(result)[0].strip() if result else ""
            pred, report = safe_execute(code)
            pred = floatify_ans(pred)
            corrected = True
            print("{}\n```".format(code))
            print("Execution:", report)
            print("Output: answer =", pred)

            if code.strip() == sample['code'][base_idx].strip(): # no correction
                corrected = False
                code = sample['code'][base_idx]
                report = sample['report'][base_idx]
                pred = sample['pred'][base_idx]

            # append new result
            sample['code'].append(code)
            sample['report'].append(report)
            sample['pred'].append(pred)
            is_correct = finqa_equal(pred, sample['gt'])

            print("Gold Answer:", sample['gt'])
            print("Corrected:", "Yes" if corrected else "No")
            print("Is correct:", is_correct)
        
        writer.write(json.dumps(sample) + '\n')
        writer.flush()

    writer.close()


if __name__ == "__main__":
    args = parse_args()
    set_seed((args.seed))
    critic(args)
