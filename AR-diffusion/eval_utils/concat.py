from tqdm import tqdm

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_gpu', default="8", type=int)
parser.add_argument('--num', default="0", type=int)
parser.add_argument('--seed', default="101", type=int)
parser.add_argument('--src_path', default="", type=str)
parser.add_argument('--tgt_path', default="", type=str)
args = parser.parse_args()

gen_name = f'gen_data/gen{args.num}.txt'
if not os.path.exists(os.path.join(args.tgt_path, 'gen_data')):
    os.mkdir(os.path.join(args.tgt_path, 'gen_data'))
if os.path.exists(os.path.join(args.tgt_path, gen_name)):
    os.remove(os.path.join(args.tgt_path, gen_name))

wp = open(os.path.join(args.tgt_path, gen_name), 'w')
for i in range(args.n_gpu):
    gen_text_path = os.path.join(args.src_path, "rank" + str(i)+f"_seed_{args.seed}.txt")
    with open(gen_text_path, "r") as ifile:
        for line in tqdm(ifile):
            wp.write(line)
wp.close()
ifile.close()
