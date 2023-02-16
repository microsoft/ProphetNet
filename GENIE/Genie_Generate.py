import os
from util import logger
from train_util import dist_util
from util.util import (
    create_model_and_diffusion,
    args_to_dict,
)
# from transformers import set_seed
import torch
import collections
import argparse
from transformers import AutoTokenizer
import numpy as np
from functools import partial
from data_util.s2s_data_util import load_s2s_data
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from data_util.s2s_data_util import S2S_dataset, QG_dataset_Diff
from torch.serialization import default_restore_location


from transformers import (
    BertModel,
    BertConfig,
    AutoTokenizer,
)
from data_util.text_data_util import load_data_text
from tqdm import tqdm
import random


def get_arguments():
    parser = argparse.ArgumentParser()

    # out path
    parser.add_argument('--generate_path', type=str, default='', help='output path')
    parser.add_argument('--eval_model_path', type=str, default='', help='model path')
    parser.add_argument('--num_samples', type=int, default=50, help='sample query')
    parser.add_argument('--interval_step', type=int, default=1, help='inference t interval step')

    # load model
    parser.add_argument('--model_arch', type=str, default='transformer', help='Core architecture of diffusion model')
    parser.add_argument('--model_channels', type=int, default=768,
                        help='Try to set it to the same size as the model hidden')
    parser.add_argument('--in_channel', type=int, default=768,
                        help='The input chanel size here must be the same as the word embedding size')
    parser.add_argument('--out_channel', type=int, default=768,
                        help='The dimension size of the output is recommended to be the same as that of word embedding for easy reasoning')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument("--learn_sigma", default=False, action="store_true", help="Whether to learning variance")
    parser.add_argument('--logits_mode', type=int, default=1, help='final logits mode of Diffusion model')
    parser.add_argument('--vocab_size', type=int, default=30522, help='vocab size')
    parser.add_argument('--config_name', type=str, default='bert-base-uncased', help='')
    parser.add_argument('--token_emb_type', type=str, default='random', help='token embedding type')
    parser.add_argument("--init_pretrained", default=False, action="store_true",
                        help="Whether to using pretrain BERT encoder")

    # load diffusion
    # parser.add_argument('--model_arch', type=str, default='transformer', help='Core architecture of diffusion model')
    parser.add_argument('--diffusion_steps', type=int, default=2000, help='Diffusion model maximum T')
    # parser.add_argument("--learn_sigma", default=False, action="store_true", help="Whether to learning variance")
    parser.add_argument('--use_kl', default=False, action="store_true",
                        help="Whether to using kl loss in Diffsion loss")
    parser.add_argument('--training_mode', type=str, default='e2e', help='using e2e simple loss or e2e loss')
    parser.add_argument('--noise_schedule', type=str, default='sqrt',
                        help='How to plan the noise change of Gaussian distribution')
    parser.add_argument('--predict_xstart', default=False, action="store_true",
                        help="Model prediction target, if True, predict xstart, if False, predict EPSILON")
    parser.add_argument("--sigma_small", default=False, action="store_true", help="about learning variance")
    parser.add_argument("--rescale_learned_sigmas", default=True, action="store_false", help="about learning variance")
    parser.add_argument("--rescale_timesteps", default=True, action="store_false", help="about time rescale")

    # data args
    parser.add_argument('--data_path', type=str, default='', help='data path')
    parser.add_argument('--data_name', type=str, default='', help='data name')
    # for seq2seq
    parser.add_argument('--src_max_len', type=int, default=144, help='src max len')
    parser.add_argument('--tgt_max_len', type=int, default=32, help='tgt max len')
    parser.add_argument('--answer_max_len', type=int, default=10, help='tgt max len')

    # gen args
    parser.add_argument('--batch_size', type=int, default=64, help='')

    # seed
    parser.add_argument('--seed', type=int, default=101, help='')
    #
    # muti-gpu
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    args = parser.parse_args()
    return args


CheckpointState = collections.namedtuple("CheckpointState",
                                                     ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset'])

def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)


'''
rounding
'''
def denoised_fn_round(args, model, text_emb, t):
    # thresh_t = 50
    # # print(thresh_t)
    # if thresh_t is not None and t[0] > thresh_t:
    #     return text_emb

    if args.model_arch == '1d-unet':
        text_emb = text_emb.permute(0, 2, 1)
    # return text_emb
    # print(t.float().mean(), t[0])

    # assert t.float().mean() == t[0].float()

    # print(text_emb.shape) # bsz, seqlen, dim
    down_proj_emb = model.weight  # input_embs
    # print(t)
    old_shape = text_emb.shape
    old_device = text_emb.device

    def get_efficient_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            emb_norm = (down_proj_emb ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            # print(emb_norm.shape, arr_norm.shape)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(down_proj_emb,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            dist = torch.clamp(dist, 0.0, np.inf)
            # print(dist.shape)
        topk_out = torch.topk(-dist, k=1, dim=0)
        #     adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
        #         down_proj_emb.size(0), -1, -1)
        #     adjacency = -th.norm(adjacency, dim=-1)
        # topk_out = th.topk(adjacency, k=1, dim=0)
        # print(topk_out1.indices == topk_out.indices)
        # assert th.all(topk_out1.indices == topk_out.indices)
        return topk_out.values, topk_out.indices

    def get_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
                down_proj_emb.size(0), -1, -1)
            adjacency = -torch.norm(adjacency, dim=-1)
        topk_out = torch.topk(adjacency, k=1, dim=0)
        return topk_out.values, topk_out.indices

    dist = 'l2'
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb
    # val, indices = get_knn(down_proj_emb,
    #                        text_emb.to(down_proj_emb.device), dist=dist)
    val, indices = get_efficient_knn(down_proj_emb,
                                     text_emb.to(down_proj_emb.device), dist=dist)
    rounded_tokens = indices[0]
    # print(rounded_tokens.shape)
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)
    if args.model_arch == '1d-unet':
        new_embeds = new_embeds.permute(0, 2, 1)
    return new_embeds



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_env(args):
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

def main():
    # env setting
    args = get_arguments()
    # setup_seed(args.seed)
    setup_env(args)

    if dist.get_rank() == 0:
        if not os.path.exists(args.generate_path):
            os.makedirs(args.generate_path)

    log_path = os.path.join(args.generate_path, 'log')
    logger.configure(dir=log_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model and diffusion
    model, diffusion = create_model_and_diffusion(
        args
    )
    model.to(args.device)
    model.eval()
    # load trained model
    model_saved_state = load_states_from_checkpoint(args.eval_model_path)
    model.load_state_dict(model_saved_state.model_dict)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    if dist.get_world_size() > 1:
        model = DDP(
            model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=False,
        )

    logger.log("sampling text from random noise...")
    print("sample num is :", args.num_samples)
    print("sample interval step is :", args.interval_step)
    print("total inverse diffusion step is :", 2000 // args.interval_step)

    sample_fn = (
        diffusion.p_sample_loop
    )

    if dist.get_world_size() > 1:
        emb_model = model.module.word_embedding
    else:
        emb_model = model.word_embedding

    if args.model_arch == 'transformer':

        sample_shape = (args.num_samples, args.text_max_len, args.in_channel)
        sample = sample_fn(
            model,
            sample_shape,
            clip_denoised=False,
            denoised_fn=partial(denoised_fn_round, args, emb_model.cuda()),
            model_kwargs=None,
            top_p=-1.0,
        )
        print("sample result shape: ", sample.shape)
        print('decoding for e2e... ')

        logits = model.get_logits(sample)
        cands = torch.topk(logits, k=1, dim=-1)
        sample_id_list = cands.indices
        print("decode id list example :", type(sample_id_list[0]), "  ", sample_id_list[0])

        logger.log("creating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        for sample_id in sample_id_list:
            sentence = tokenizer.decode(sample_id.squeeze())
            print(sentence)

    elif args.model_arch == 's2s_CAT':

        # bert tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        print("-------------------------------------------------------------")
        print("start generate query from dev dataset, for every passage, we generate ", args.num_samples, " querys...")
        print("-------------------------------------------------------------")

        print("***** load " + args.data_name + " test src dataset*****")
        src = []
        test_src_path = os.path.join(args.data_path, args.data_name + "/org_data/test.src")
        with open(test_src_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                src.append(text)

        print("***** load " + args.data_name + " dev tgt dataset*****")
        tgt = []
        test_tgt_path = os.path.join(args.data_path, args.data_name + "/org_data/test.tgt")
        with open(test_tgt_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                text = line
                tgt.append(text)

        shard_size = len(src) // args.world_size
        start_idx = args.local_rank * shard_size
        end_idx = start_idx + shard_size
        if args.local_rank == args.world_size - 1:
            end_idx = len(src)
        scr_data_piece = src[start_idx:end_idx]
        tgt_data_piece = tgt[start_idx:end_idx]

        print('generation for ', len(scr_data_piece), " src text from idx ", start_idx, " to ", end_idx)
        if args.data_name == "squadqg_data":
            test_dataset = QG_dataset_Diff(scr_data_piece, tgt_data_piece, tokenizer, src_maxlength=args.src_max_len,
                                       answer_maxlength=args.answer_max_len, tgt_maxlength=args.tgt_max_len)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False,
                                         num_workers=20, collate_fn=QG_dataset_Diff.get_collate_fn())
        else:
            test_dataset = S2S_dataset(scr_data_piece, tgt_data_piece, tokenizer, src_maxlength=args.src_max_len,
                                       tgt_maxlength=args.tgt_max_len)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False,
                                          num_workers=20, collate_fn=S2S_dataset.get_collate_fn())

        if args.generate_path is not None:
            model_gen_files = []
            if os.path.exists(args.generate_path):
                for item in os.scandir(args.generate_path):
                    if item.is_file():
                        if "gen_seed" in item.path:
                            model_gen_files.append(item.path)
                if len(model_gen_files) != 0 :
                    model_gen_files.sort(key=lambda f: int((f.split('_epoch')[-1]).split('.txt')[0]), reverse=True)
                    epoch_num = int((model_gen_files[0].split('_epoch')[-1]).split('.txt')[0])
                    logger.info("***** load " + model_gen_files[0] + " *****")
                else:
                    epoch_num = 0

        else:
            logger.info("generate_path is None")
            exit(0)

        for epoch in range(args.num_samples - epoch_num):
            each_sample_list = []
            print("-------------------------------------------------------------")
            print("start sample ", epoch+1+epoch_num, " epoch...")
            print("-------------------------------------------------------------")

            for index, batch in enumerate(tqdm(test_dataloader)):
                '''
                for s2s
                '''
                input_shape = (batch['src_input_ids'].shape[0], args.tgt_max_len, args.in_channel)
                src_input_ids = batch['src_input_ids']
                tgt_input_ids = batch['tgt_input_ids']
                # print(p_input_ids.shape)
                src_attention_mask = batch['src_attention_mask']
                model_kwargs = {'src_input_ids' : src_input_ids, 'src_attention_mask': src_attention_mask}

                sample = sample_fn(
                    model,
                    input_shape,
                    clip_denoised=False,
                    denoised_fn=partial(denoised_fn_round, args, emb_model.cuda()),
                    model_kwargs=model_kwargs,
                    top_p=-1.0,
                    interval_step=args.interval_step,
                )

                print("sample result shape: ", sample.shape)
                print('decoding for e2e... ')

                logits = model.module.get_logits(sample)
                cands = torch.topk(logits, k=1, dim=-1)
                sample_id_list = cands.indices
                #print("decode id list example :", type(sample_id_list[0]), "  ", sample_id_list[0])

                '''
                for s2s
                '''
                # print("src text: ", tokenizer.decode(src_input_ids.squeeze()))
                # print("tgt text: ", tokenizer.decode(tgt_input_ids.squeeze()))

                print("sample control generate query: ")
                for sample_id in sample_id_list:
                    sentence = tokenizer.decode(sample_id.squeeze())
                    each_sample_list.append(clean(sentence))
                    # print(sentence)

            # total_sample_list.append(each_sample_list)
            out_path = os.path.join(args.generate_path, "rank" + str(dist.get_rank()) + "_gen_seed_101" +
                                    "_num" + str(args.num_samples) + "_epoch" + str(epoch + 1 + epoch_num) + ".txt")
            with open(out_path, 'w') as f:
                for sentence in each_sample_list:
                    f.write(sentence + '\n')

    else:
        return NotImplementedError

def clean(sentence):
    sentence = sentence.replace('[CLS]', '')
    sentence = sentence.replace('[SEP]', '')
    sentence = sentence.replace('[PAD]', '')
    sentence = sentence.replace('[UNK]', 'unk')
    return sentence.strip()

if __name__ == "__main__":
    main()