import json
import ast
import numpy as np
import pprint
from src.utils import load_jsonl
from src.qa.utils import multi_ref_score, is_null_answer, extract_cot_answer


def rejection_sampling_eval(file_path, n=10):
    all_scores = []
    for idx, sample in enumerate(load_jsonl(file_path)):
        if "prediction" not in sample or "temperature_0.5" not in sample['prediction']:
            continue
        sampled_preds = sample['prediction']['temperature_0.5']['text'][:n]

        # extract cot answer
        if "cot" in file_path:
            sampled_preds = [extract_cot_answer(p) for p in sampled_preds]
        
        scores = [multi_ref_score(pred, sample['answer']) for pred in sampled_preds]

        # get max scores: best-of-n
        best_score = max(scores)
        all_scores.append(best_score)

    # average score of all samples
    average_score = np.array(all_scores).mean(axis=0)
    em, f1 = list(np.round(average_score * 100, decimals=1))
    print(f"Best-of-{n}: {em} & {f1}")


def evaluate(file_path, oracle=False, verbose=True, max_iter=4, critic=True):
    if verbose:
        print(file_path)

    em_scores = []
    f1_scores = []

    for idx, sample in enumerate(load_jsonl(file_path)):
        
        if 'pred' not in sample:
            if "cot" in file_path:
                sample['pred'] = [extract_cot_answer(sample['prediction']['greedy']['text'])]
            elif "direct" in file_path:
                if 'greedy' not in sample['prediction']: # jump failed direct answer
                    continue
                sample['pred'] = [sample['prediction']['greedy']['text']]

        cur_em = []
        cur_f1 = []

        for itr in range(max_iter):

            # stopped
            if itr > len(sample['pred']) - 1:
                cur_em.append(cur_em[-1])
                cur_f1.append(cur_f1[-1])
                continue

            # the latest not NULL pred
            if critic:
                for j in range(itr, -1, -1):
                    if not is_null_answer(sample['pred'][j]):
                        break
                pred = sample['pred'][j]
            else:
                pred = sample['pred'][itr]

            em, f1 = multi_ref_score(pred, sample['answer'])
            cur_em.append(em)
            cur_f1.append(f1)

            # early stop
            stop = (sample['pred'][itr] == sample['pred'][itr - 1])

            if (oracle and em) or (not oracle and stop):
                cur_em.extend([em] * (max_iter - itr - 1))
                cur_f1.extend([f1] * (max_iter - itr - 1))
                break

        em_scores.append(cur_em)
        f1_scores.append(cur_f1)

    # output mean of each column of scores
    em_means = np.array(em_scores).mean(axis=0)
    em_means = list(np.round(em_means* 100, decimals=1))

    f1_means = np.array(f1_scores).mean(axis=0)
    f1_means = list(np.round(f1_means* 100, decimals=1))

    if verbose:
        print("num of samples:", len(em_scores))
        print(em_means)
        print(f1_means)
        print(f"CoT EM/F1:\t{em_means[0]} & {f1_means[0]}")
        print("CRITIC (oracle)" if oracle else "CRITIC", end=" ")
        print(f"EM/F1:\t{em_means[-1]} & {f1_means[-1]}\n")

    return em_means, f1_means


if __name__ == "__main__":

    ## text-davinci-003
    # critic
    # file_path = "outputs/text-davinci-003/ambig_qa/validation_critic_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/trivia_qa/validation_critic_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/hotpot_qa/validation_critic_500_seed0.jsonl"

    # critic no tools
    # file_path = "outputs/text-davinci-003/ambig_qa/validation_critic_no-tool_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/trivia_qa/validation_critic_no-tool_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/hotpot_qa/validation_critic_no-tool_500_seed0.jsonl"

    # direct
    # file_path = "outputs/text-davinci-003/ambig_qa/validation_direct_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/trivia_qa/validation_direct_500_seed0.jsonl"
    # file_path = "outputs/text-davinci-003/hotpot_qa/validation_direct_500_seed0.jsonl"

    ## gpt-3.5-turbo
    # critic
    # file_path = "outputs/gpt-3.5-turbo/ambig_qa/validation_critic_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/trivia_qa/validation_critic_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/hotpot_qa/validation_critic_500_seed0.jsonl"

    # critic no tools
    # file_path = "outputs/gpt-3.5-turbo/ambig_qa/validation_critic_no-tool_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/trivia_qa/validation_critic_no-tool_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/hotpot_qa/validation_critic_no-tool_500_seed0.jsonl"

    # direct
    # file_path = "outputs/gpt-3.5-turbo/ambig_qa/validation_direct_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/trivia_qa/validation_direct_500_seed0.jsonl"
    # file_path = "outputs/gpt-3.5-turbo/hotpot_qa/validation_direct_500_seed0.jsonl"

    # evaluate(file_path)
    # evaluate(file_path, oracle=True)
    # exit()

    ## react
    # for file_path in [
    #     ## gpt-3.5-turbo
    #     "outputs/gpt-3.5-turbo/ambig_qa/validation_react_500_seed0.jsonl",
    #     "outputs/gpt-3.5-turbo/trivia_qa/validation_react_500_seed0.jsonl",
    #     "outputs/gpt-3.5-turbo/hotpot_qa/validation_react_500_seed0.jsonl",

    #     ## text-davinci-003
    #     "outputs/text-davinci-003/ambig_qa/validation_react_500_seed0.jsonl",
    #     "outputs/text-davinci-003/trivia_qa/validation_react_500_seed0.jsonl",
    #     "outputs/text-davinci-003/hotpot_qa/validation_react_500_seed0.jsonl",
    # ]:
    #     evaluate(file_path, max_iter=2, critic=False)
    # exit()


    ## rejection sampling: best-of-n
    # for file_path in [
    #     "outputs/text-davinci-003/ambig_qa/validation_cot_500_seed0_t0.5.jsonl",
    #     "outputs/text-davinci-003/trivia_qa/validation_cot_500_seed0_t0.5.jsonl",
    #     "outputs/text-davinci-003/hotpot_qa/validation_cot_500_seed0_t0.5.jsonl",
    #     "outputs/gpt-3.5-turbo/ambig_qa/validation_cot_500_seed0_t0.5.jsonl",
    #     "outputs/gpt-3.5-turbo/trivia_qa/validation_cot_500_seed0_t0.5.jsonl",
    #     "outputs/gpt-3.5-turbo/hotpot_qa/validation_cot_500_seed0_t0.5.jsonl",
    # ]:
    #     print(file_path)
    #     rejection_sampling_eval(file_path, n=4)
    #     print()
    # exit()
    
    # all in one
    metrics = {}
    for data in ['ambig_qa', 'trivia_qa', "hotpot_qa"]:
        metrics[data] = {}
        for model in ['gpt-3.5-turbo', 'text-davinci-003']:
            metrics[data][model] = {}

            # critic
            file_path = f"outputs/{model}/{data}/validation_critic_500_seed0.jsonl"
            em, f1 = evaluate(file_path, verbose=False)

            # critic(oracle)
            em_oracle, f1_oracle = evaluate(file_path, oracle=True, verbose=False)

            # critic w/o tool
            file_path = f"outputs/{model}/{data}/validation_critic_no-tool_500_seed0.jsonl"
            em_notool, f1_notool = evaluate(file_path, verbose=False)

            # react
            file_path = f"outputs/{model}/{data}/validation_react_500_seed0_t0.jsonl"
            em_react , f1_react = evaluate(file_path, max_iter=2, critic=False, verbose=False)
            
            # save em and f1 to metrics
            metrics[data][model]['em'] = em
            metrics[data][model]['em_oracle'] = em_oracle
            metrics[data][model]['em_notool'] = em_notool
            metrics[data][model]['em_react'] = em_react
            metrics[data][model]['f1'] = f1
            metrics[data][model]['f1_oracle'] = f1_oracle
            metrics[data][model]['f1_notool'] = f1_notool
            metrics[data][model]['f1_react'] = f1_react

    pprint.pprint(metrics, width=160)
