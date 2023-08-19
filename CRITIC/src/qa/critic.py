import sys
import re
import json
from datetime import datetime
import random
import time
import requests
import pprint
import numpy as np
import argparse
from collections import Counter
from distutils.util import strtobool

from src.datasets.dataset_loader import DatasetLoader
from src.llms.api import llm
from src.utils import set_seed, load_jsonl, list_rindex, load_prompt
from src.qa.utils import em_f1_score, multi_ref_score, extract_cot_answer, get_end_index, is_null_answer
from src.tools.search_api import google


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="hotpot_qa", type=str)
    parser.add_argument("--model", default="text-davinci-003", type=str)
    parser.add_argument("--critic_type", default="critic", type=str)
    parser.add_argument("--prompt_type", default="cot", type=str)
    parser.add_argument("--split", default="validation", type=str)
    parser.add_argument("--num_test_sample", default=500, type=int) # -1 for full data
    parser.add_argument("--max_tokens", default=300, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--max_iter", default=3, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--use_tool", type=strtobool, default=True)
    parser.add_argument("--evidence_length", default=400, type=int)
    parser.add_argument("--max_interaction", default=7, type=int)
    args = parser.parse_args()
    return args


def critic_iter(sample, previous_cot, args):

    # load prompt
    prompt_critic = load_prompt(args.data, args.critic_type)

    # construct context
    context = f"Question: {sample['question']}\nProposed Answer: {previous_cot}\n\n"

    # verify: plausible & truthful
    context += "What's the problem with the above answer?\n\n1. Plausibility:\n\n"
    print(context, end="")
    prompt_critic += context

    exist_query = []
    exist_evidence = set()
    revised_cot = ""
    for idx in range(args.max_interaction): # max interaction with tool
        # get LLM res
        res = ""
        res = llm(prompt=prompt_critic,
                    model=args.model,
                    stop=["> Evidence:", "---"],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    verbose=False)
        if not res:
            res = ""
        res = res['choices'][0]['text']
        res = res.lstrip(" ).:\n") # fix bugs in API

        # case1: search
        if "> Search Query:" in res:
            try:
                _, search_query = res.split("> Search Query:")[:2]
                search_query = search_query.split("\n")[0].strip()
            except:
                print("Search Query Error:", res)
                exit()

            context = res
            print(context, end="")
            prompt_critic += context

            if args.use_tool:
                # use Tool: search a new evidence
                exist_query.append(search_query)
                for k in range(exist_query.count(search_query), 8):
                    search_res = google(search_query, topk=k)
                    if search_res['page'] not in exist_evidence:
                        exist_evidence.add(search_res['page'])
                        break

                context = f"""> Evidence: [{search_res['title']}] {search_res['page'][:args.evidence_length]}\n\n"""
                if idx == arg.max_interaction - 2:
                    context += f"Let's give the most possible answer.\n\nQuestion: {sample['question']}\nHere's "
            else:
                # w/o Tool: use LLMs generated evidence
                context = """> Evidence: """
            print(context, end="")
            prompt_critic += context
            
        # case2: most possible answer
        elif "most possible answer: " in res:
            print(res)
            _, revised_cot = res.split("most possible answer: ")
            revised_cot = revised_cot.strip()
            break
        # case3: other output
        else:
            if not res:
                break
            context = res
            context += f"Let's give the most possible answer.\n\nQuestion: {sample['question']}\nHere's "
            print(context, end="")
            prompt_critic += context

    return revised_cot


def critic(args):
    # input and output file
    now = datetime.now()
    dt_string = now.strftime("%m-%d_%H-%M")
    assert args.prompt_type == "cot"
    init_file = f"outputs/{args.model}/{args.data}/{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}.jsonl"
    out_file = f'outputs/{args.model}/{args.data}/{args.split}_{args.critic_type}{"" if args.use_tool else "_no-tool"}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}_s{args.start}_e{args.end}_{dt_string}.jsonl'

    writer = open(out_file, 'w')

    for idx, sample in enumerate(load_jsonl(init_file)):
        if idx < args.start or (args.end != -1 and idx >= args.end):
            continue

        # if empty result, continue
        if "greedy" not in sample['prediction']:
            continue

        # add idx to the begining of sample
        if 'idx' not in sample:
            sample = {**{'idx': idx}, **sample}

        print("\n\n" + "=" * 30, "Idx", idx, "=" * 30)
        print(f"Question: {sample['question']}")
        print(f"Gold answer: {sample['answer']}")
       
        # iterative correction
        previous_corrected = True
        for itr in range(1, args.max_iter + 1):
            # initializaiton
            if itr == 1:
                # extract prediction
                init_cot = sample['prediction']['greedy']['text']
                init_pred = extract_cot_answer(init_cot)
                print(f"Init pred: {init_pred}")
                em, f1 = multi_ref_score(init_pred, sample['answer'])
                print(f"EM/F1: {em}/{f1:.2f}")

                # cot and pred
                sample['cot'] = [init_cot]
                sample['pred'] = [init_pred]
                sample.pop('prediction')

            print("\n" + "-" * 20, "iteration", itr, "-" * 20)

            # choose the latest answer that is not "None" to critic
            base_idx = itr - 1
            while base_idx > 0 and is_null_answer(sample['pred'][base_idx]):
                base_idx -= 1
            previous_cot = sample['cot'][base_idx]
            previous_pred = sample['pred'][base_idx]
            print("Base Iter:", base_idx)

            # one iteration
            revised_cot = critic_iter(sample, previous_cot, args)
            revised_pred = extract_cot_answer(revised_cot)

            # is corrected
            corrected = True
            if revised_cot and (revised_cot == previous_cot):
                corrected = False
            
            if is_null_answer(revised_pred):
                print(">>> Null Answer")

            # get metric
            em, f1 = multi_ref_score(revised_pred, sample['answer'])
            print(f"Revised pred: {revised_pred}")
            print(f"Gold answer: {sample['answer']}")
            print(f"Corrected: {'Yes' if corrected else 'No'}")
            print(f"EM/F1: {em}/{f1:.2f}")

            sample['cot'].append(revised_cot)
            sample['pred'].append(revised_pred)

            # if no correction for twice, break
            if not corrected and not previous_corrected:
                print("Stop.")
                break
            previous_corrected = corrected
   
        writer.write(json.dumps(sample) + '\n')
        writer.flush()

    writer.close()


if __name__ == "__main__":
    args = parse_args()
    set_seed((args.seed))
    critic(args)
