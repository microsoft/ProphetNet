# developed based on: https://github.com/GXimingLu/Quark/blob/main/perplexity.py
import json
import numpy as np
from tqdm import tqdm
from src.llms.api import llm


def calc_ppl(in_file, model, total_iter=5, start=0, end=-1):
    all_ppl = []

    with open(in_file, 'r') as f:
        data = json.load(f)

    for iter in range(total_iter):
        print("-" * 30, "Iter", iter)
        iter_ppls = []
        samples = data[str(iter)]

        for idx, sample in tqdm(enumerate(samples)):
            if idx < start or (end != -1 and idx >= end):
                continue

            prompt = sample['prompt'].lstrip()
            prompt_logprobs = get_logprobs(model, prompt)
            if not prompt_logprobs:
                continue
            prompt_tokens = prompt_logprobs['tokens']

            for gen in sample['continuation']:
                gen_split = gen.lstrip().split(" ")
                gen = handle_strip_spaces(prompt, gen_split)
                prompt_continuation = prompt + gen
                gen_logprobs = get_logprobs(model, prompt_continuation)['token_logprobs'][len(prompt_tokens):]

                ppl = calculate_ppl(gen_logprobs)
                if ppl < 1e3: # for sanity
                    iter_ppls.append(ppl)

        all_ppl.append(np.nanmean(iter_ppls))
        print(all_ppl)

    out_file = in_file.replace(".json", "_ppl.txt")
    with open(out_file, 'a') as fo:
        fo.write(f's_{start}_e{end}_perplexity = {all_ppl}\n')
    return all_ppl


def get_logprobs(model, prompt):
    response = llm(model=model, prompt=prompt, max_tokens=1, logprobs=1, n=1, stop=None, temperature=0, echo=True)
    try:
        logprobs = response['choices'][0]['logprobs']
    except:
        print(">" * 20)
        return None
    return logprobs['tokens']


def handle_strip_spaces(prompt, gen_split):
    if prompt.strip().endswith(gen_split[0]):
        gen_split = gen_split[1:]
    gen = " ".join(gen_split)
    if not prompt.endswith(" ") and not gen.startswith(" "):
        gen = " " + gen
    if prompt.endswith(" ") and gen.startswith(" "):
        gen = gen[1:]
    return gen


def calculate_ppl(gen_logprobs):
    return np.exp(-np.mean(gen_logprobs))


if __name__ == "__main__":

    model = "text-davinci-003"
    in_file = "outputs/text-davinci-003/toxicity/test_critic_1000_seed0_all_samples.json"
    # in_file = "outputs/text-davinci-003/toxicity/test_critic_no-tool_1000_seed0_all_samples.json"

    # in_file = "outputs/gpt-3.5-turbo/toxicity/test_critic_1000_seed0_all_samples.json"
    # in_file = "outputs/gpt-3.5-turbo/toxicity/test_critic_no-tool_1000_seed0_all_samples.json"

    calc_ppl(in_file, model, start=0, end=1000)
