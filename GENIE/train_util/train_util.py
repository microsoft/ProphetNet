import torch
import copy
import os
from torch import nn
import collections
from util import logger
from train_util import dist_util
from transformers import AdamW
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
from torch.serialization import default_restore_location
from transformers import (
    get_linear_schedule_with_warmup,
)
from transformers import (
    BertModel,
    BertConfig,
)
from diffusion_util.resample import LossAwareSampler, UniformSampler

from data_util.text_data_util import Text_Hidden_dataset, Question_dataset, PandQ_dataset
from data_util.s2s_data_util import S2S_dataset, QG_dataset_Diff

INITIAL_LOG_LOSS_SCALE = 20.0
CheckpointState = collections.namedtuple("CheckpointState",
                                                     ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset'])

'''
TrainLoop training class
'''
class TrainLoop:
    def __init__(
        self,
        train_type,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        warmup_steps=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        checkpoint_path='',
        gradient_clipping=-1.,
        eval_data=None,
        eval_interval=-1,
        gradient_accumulation_steps=1,
        device=None,
        data_name="xsum_data",
    ):
        self.train_type = train_type
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.warmup_steps = warmup_steps
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.gradient_clipping = gradient_clipping
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.data_name = data_name
        self.master_params = list(self.model.parameters())
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

        self.checkpoint_path = checkpoint_path
        self.optimizer = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.lr_anneal_steps
        )
        self.global_step = 0

        # load last checkpoint
        if self.checkpoint_path is not None:
            model_checkpoint_files = []
            ema_checkpoint_files = []
            if os.path.exists(self.checkpoint_path):
                for item in os.scandir(self.checkpoint_path):
                    if item.is_file():
                        if "model_checkpoint" in item.path:
                            model_checkpoint_files.append(item.path)
                        if "ema" in item.path:
                            ema_checkpoint_files.append(item.path)
                if len(model_checkpoint_files) != 0 and len(ema_checkpoint_files) != 0:
                    model_checkpoint_files.sort(key=lambda f: int(f.split('model_checkpoint-')[1]), reverse=True)
                    logger.info("***** load " + model_checkpoint_files[0] + " *****")
                    ema_checkpoint_files.sort(key=lambda f: int(f.split('checkpoint-')[-1]), reverse=True)
                    logger.info("***** load " + ema_checkpoint_files[0] + " *****")
                    model_saved_state = load_states_from_checkpoint(model_checkpoint_files[0])
                    self.global_step = self._load_saved_state(model_saved_state)
                    self.ema_params = [
                        copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
                    ]
                else:
                    self.ema_params = [
                        copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
                    ]
                    logger.info("***** there are no checkpoint in" + self.checkpoint_path + " *****")

        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        # model to DDP
        if dist.get_world_size() > 1:
            self.model = DDP(
                self.model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=False,
            )
        else:
            print("single GPU is not achieve now")
            exit(0)

    def run_loop(self):
        logger.info("***** Running training *****")
        logger.info("  Max steps = %d", self.lr_anneal_steps)
        logger.info("  Instantaneous batch size per GPU = %d", self.batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.batch_size
            * self.gradient_accumulation_steps
            * (dist.get_world_size()),
        )
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        self.model.zero_grad()
        self.model.train()

        # ddp data sample
        if self.train_type == 'LM_Diffusion':
            train_sample = DistributedSampler(self.data)
            train_dataloader = DataLoader(self.data, sampler=train_sample, batch_size=self.batch_size, drop_last=False,
                                          num_workers=20, collate_fn=Question_dataset.get_collate_fn())
        elif self.train_type == 'S2S_Diffusion':
            train_sample = DistributedSampler(self.data)
            '''
            for s2s
            '''
            train_dataloader = DataLoader(self.data, sampler=train_sample, batch_size=self.batch_size, drop_last=False,
                                      num_workers=20, collate_fn=S2S_dataset.get_collate_fn())
        else:
            return NotImplementedError

        while self.global_step < self.lr_anneal_steps:

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=dist.get_rank() not in [-1, 0])

            for batch in epoch_iterator:
                self.model.train()

                # forward loss
                self.forward_backward(batch)
                if self.use_fp16:
                    pass
                else:
                    # gradient clip
                    if self.gradient_clipping > 0:
                        self.grad_clip()
                    self._log_grad_norm()
                    self.optimizer.step()
                    # lr scheduler
                    self.scheduler.step()
                    self.model.zero_grad()
                    # ema
                    for rate, params in zip(self.ema_rate, self.ema_params):
                        self.update_ema(params, self.master_params, rate=rate)
                self.global_step += 1
                self.log_step()

                if self.global_step % self.log_interval == 0:
                    logger.dumpkvs()

                if self.eval_data is not None and self.global_step % self.eval_interval == 0:
                    if dist.get_rank() == 0:
                        print('eval on validation set...')
                        if self.train_type == 'LM_Diffusion':
                            dev_dataloader = DataLoader(self.eval_data, batch_size=self.batch_size,
                                                          drop_last=False,
                                                          num_workers=20, collate_fn=Question_dataset.get_collate_fn())
                        elif self.train_type == 'S2S_Diffusion':
                            '''
                            for s2s
                            '''
                            dev_dataloader = DataLoader(self.eval_data, batch_size=self.batch_size,
                                                        drop_last=False,
                                                        num_workers=20, collate_fn=S2S_dataset.get_collate_fn())
                        else:
                            return NotImplementedError
                        for step, batch in enumerate(dev_dataloader):
                            self.forward_only(batch)
                            if step > 10:
                                break
                        logger.dumpkvs()
                # save
                if self.global_step % self.save_interval == 0:
                    self.save()

    def save(self):

        def save_checkpoint(rate, ema_params):
            model_to_save = get_model_obj(self.model)
            if not rate:
                model_state_dict = model_to_save.state_dict()
            else:
                model_state_dict = model_to_save.state_dict()
                for i, (name, _value) in enumerate(model_to_save.named_parameters()):
                    assert name in model_state_dict
                    model_state_dict[name] = ema_params[i]

            opt_state_dict = self.optimizer.state_dict()
            sch_state_dict = self.scheduler.state_dict()
            offset = self.global_step
            state = CheckpointState(model_state_dict,
                                    opt_state_dict,
                                    sch_state_dict,
                                    offset,
                                    )
            if not rate:
                ckpt_path = os.path.join(self.checkpoint_path, 'model_checkpoint-' + str(offset))
            else:
                ckpt_path = os.path.join(self.checkpoint_path, 'ema_' + str(rate) + '_checkpoint-' + str(offset))

            torch.save(state._asdict(), ckpt_path)
            logger.info('Saved checkpoint at %s', ckpt_path)

        if dist.get_rank() == 0:
            save_checkpoint(0, None)
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params)


    def forward_backward(self, batch):

        if self.train_type == 'LM_Diffusion':
            t, weights = self.schedule_sampler.sample(batch[1].shape[0], self.device)
            inputs_text = {"query_ids": batch[1].long().to(self.device),
                                "attention_mask_q": batch[2].long().to(self.device)}
            losses = self.diffusion.training_losses(self.model, inputs_text, t)
        elif self.train_type == 'S2S_Diffusion':

            '''
            for s2s
            '''
            t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], self.device)
            losses = self.diffusion.training_losses(self.model, batch, t)
        else:
            return NotImplementedError

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )

        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def forward_only(self, batch):
        with torch.no_grad():
            self.model.zero_grad()
            if self.train_type == 'LM_Diffusion':
                t, weights = self.schedule_sampler.sample(batch[1].shape[0], dist_util.dev())
                inputs_text = {"query_ids": batch[1].long().to(self.device),
                               "attention_mask_q": batch[2].long().to(self.device)}

                losses = self.diffusion.training_losses(self.model, inputs_text, t)
            elif self.train_type == 'S2S_Diffusion':

                '''
                for s2s
                '''
                t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], self.device)
                losses = self.diffusion.training_losses(self.model, batch, t)
            else:
                return NotImplementedError

            log_loss_dict(
                self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()}
            )

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            # print(p)
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def log_step(self):
        logger.logkv("step", self.global_step)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.global_step / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def grad_clip(self):
        # print('doing gradient clipping')
        max_grad_norm=self.gradient_clipping #3.0
        if hasattr(self.optimizer, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.optimizer.clip_grad_norm(max_grad_norm)
        # else:
        #     assert False
        # elif hasattr(self.model, "clip_grad_norm_"):
        #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
        #     self.model.clip_grad_norm_(args.max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), #amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )

    def _load_saved_state(self, saved_state: CheckpointState):
        self.global_step = saved_state.offset
        logger.info('Loading checkpoint @ step=%s', self.global_step)

        logger.info('Loading saved model state ...')
        self.model.load_state_dict(saved_state.model_dict)  # set strict=False if you use extra projection
        self.optimizer.load_state_dict(saved_state.optimizer_dict)
        self.scheduler.load_state_dict(saved_state.scheduler_dict)
        self.master_params = list(self.model.parameters())
        return self.global_step


    def update_ema(self, target_params, source_params, rate=0.99):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.

        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            # print("target_params:", targ.device)
            # print("source_params:", src.device)
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)



def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model

def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)



