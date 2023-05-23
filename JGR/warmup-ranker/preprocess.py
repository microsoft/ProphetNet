import json
from rouge_score import rouge_scorer, scoring
from transformers import RobertaTokenizerFast, BartTokenizerFast
import nltk
import os
from tqdm import tqdm
import argparse
from data_utils.metric_utils import compute_coqa, compute_dialog, compute_qg, compute_rouge

nltk.download('punkt')
nltk.download("omw-1.4", quiet=True)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--candidate_dir", type=str)
parser.add_argument("--save_name", type=str)
parser.add_argument("--num_cand", type=int)
parser.add_argument("--tokenizer_dir", type=str)
parser.add_argument("--tokenizer_type", type=str)
#parser.add_argument("--golden", type=str, help="Gold output file.")
args = parser.parse_args()

def generate_data(candidate_dir, data_dir, split, tokenizer,  num_cand, compute_metrics):
    '''
    args: 
        candidate_dir: the path where stores the candiddates
        data_dir: the path where stores the dataset
        compute_metrics: we will use this to generate candidates
    the raw data should be like:
        [
            {
                'source':....,
                'target':....,
            },
            ...
        ]
    the candidate file should be like one candidate per line, and the total number of lines should be equal to len(raw_data) * num_cand
    '''
    with open(os.path.join(data_dir, '%s_data.json'%(split)), encoding='utf-8') as f:
            raw_data = json.load(f)

    candidates = []
    with open(candidate_dir + '/%s/generated_predictions_%d.txt'%(split, num_cand), encoding='utf-8') as f:
        for line in f.readlines():
            candidates.append(line.strip())
    # finished here
    cur_line = 0
    samples = []

    i = 0
    for d in tqdm(raw_data):
        article = d['source']
        article_tok = tokenizer.tokenize(article)
        target = d['target']
        target_tok = tokenizer.tokenize(target)

        candidates_this = candidates[i*num_cand: (i+1)*num_cand]
        _, candidates_this,_ = compute_metrics.get_candidates([target], candidates_this, num_cand, num_cand, 'bottom')
        candidates_this = candidates_this
        cand_list = candidates_this
        cand_list_tok = [tokenizer.tokenize(c) for c in cand_list]

        sample = {
            'source': article_tok,
            'source_untok': article,
            'target': target_tok,
            'target_untok': target,
            'candidates': cand_list_tok,
            'candidates_untok': cand_list
        }
        samples.append(sample)
        i += 1


    #     with open('data/%s/%s/gen_from_%d/%d.json'%(dataset_name, split, num_cand, i), 'w', encoding='utf-8') as f:
    #         json.dump(sample, f)
    #     i += 1
    # with open('data/%s/%s/gen_from_%d/all.json'%(dataset_name, split, num_cand), 'w', encoding='utf-8') as f:
    #         json.dump(samples, f)

    return samples
        




if __name__ == "__main__":
    compute_metrics = None
    if args.dataset_name in ["cnndm", "cnndm_debug", "samsum", "xsum"]:
        compute_metrics = compute_rouge()
    elif args.dataset_name in ['squadqg']:
        compute_metrics = compute_qg()
    elif args.dataset_name in ['coqa']:
        compute_metrics = compute_coqa()
    elif args.dataset_name in ['personachat']:
        compute_metrics = compute_dialog()

    assert compute_metrics is not None

    assert args.tokenizer_type in ['roberta', 'bart']
    if args.tokenizer_type == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_dir)
    else:
        tokenizer = BartTokenizerFast.from_pretrained(args.tokenizer_dir)

    train_samples = generate_data(args.candidate_dir, args.data_dir, 'train',  tokenizer,  args.num_cand, compute_metrics)

    if not os.path.exists('data/%s/train/gen_from_%s'%(args.save_name, args.num_cand)):
        os.makedirs('data/%s/train/gen_from_%s'%(args.save_name, args.num_cand))
    with open('data/%s/train/gen_from_%s/all.json'%(args.save_name, args.num_cand), 'w', encoding='utf-8') as f:
        json.dump(train_samples, f)

    dev_samples = generate_data(args.candidate_dir, args.data_dir, 'dev',  tokenizer,  args.num_cand, compute_metrics)

    if not os.path.exists('data/%s/dev/gen_from_%s'%(args.save_name, args.num_cand)):
        os.makedirs('data/%s/dev/gen_from_%s'%(args.save_name, args.num_cand))
    with open('data/%s/dev/gen_from_%s/all.json'%(args.save_name, args.num_cand), 'w', encoding='utf-8') as f:
        json.dump(dev_samples, f)


    test_samples = generate_data(args.candidate_dir, args.data_dir, 'test',  tokenizer,  args.num_cand, compute_metrics)

    if not os.path.exists('data/%s/test/gen_from_%s'%(args.save_name, args.num_cand)):
        os.makedirs('data/%s/test/gen_from_%s'%(args.save_name, args.num_cand))
    with open('data/%s/test/gen_from_%s/all.json'%(args.save_name, args.num_cand), 'w', encoding='utf-8') as f:
        json.dump(test_samples, f)



 