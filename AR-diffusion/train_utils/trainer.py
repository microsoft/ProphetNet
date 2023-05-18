import torch
import copy
import os
import logging
import collections

import torch.distributed as dist

from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModel

from utils import get_group_parameters, load_states_from_checkpoint


logger = logging.getLogger(__name__)
CheckpointState = collections.namedtuple(
    "CheckpointState", ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset'])


class TrainLoop:
    def __init__(
        self,
        config,
        model,
        diffusion,
        data,
        dev_data,
        schedule_sampler,
    ):
        self.config = config
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.dev_data = dev_data
        self.schedule_sampler = schedule_sampler
        
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.ema_rate = (
            [config.ema_rate]
            if isinstance(config.ema_rate, float)
            else [float(x) for x in config.ema_rate.split(",")]
        )
        self.log_interval = config.log_interval
        self.eval_interval = config.eval_interval
        self.save_interval = config.save_interval
        self.warmup_steps = config.warmup_steps
        self.weight_decay = config.weight_decay
        self.total_steps = config.total_steps
        self.gradient_clipping = config.grad_clip
        self.gradient_accumulation_steps = config.grad_accum
        self.device = config.device
        self.train_type = config.model.mode

        self.master_params = list(self.model.parameters())
        self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        self.checkpoint_path = config.exp.dir
        self.writer = SummaryWriter(log_dir=self.checkpoint_path + '/board')
        
        if self.config.use_AMP:
            self.scaler = GradScaler()

        if config.load_bart:
            self.optimizer = AdamW(get_group_parameters(config, self.model))
        else:
            self.optimizer = AdamW(
                self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        
        if config.data.name == 'commongen':
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.warmup_steps
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.warmup_steps, 
                num_training_steps=int(config.lr_step)
            )
        self.global_step = 0

        # auto load last checkpoint
        self.model_path = os.path.join(self.checkpoint_path, 'model')
        if (dist.get_rank() == 0) and (not os.path.exists(self.model_path)):
            os.mkdir(self.model_path)
        dist.barrier()
        if config.resume_checkpoint:
            self.check_load()

        # model to DDP
        if dist.get_world_size() > 1:
            self.model = DDP(
                self.model, device_ids=[dist.get_rank()], 
                output_device=dist.get_rank(), find_unused_parameters=False,
            )
        else:
            print("single GPU is not achieve now")
            exit(0)

        if config.fix_encoder:
            model_cfg = AutoConfig.from_pretrained(config.model.name)
            self.encoder = AutoModel.from_pretrained(config.model.name, config=model_cfg)
            if config.load_bart:
                self.encoder = self.encoder.encoder
            self.encoder.to(self.device)
            self.encoder = DDP(
                self.encoder, device_ids=[dist.get_rank()], 
                output_device=dist.get_rank(), find_unused_parameters=False,
            )
        else:
            self.encoder = None

    def run_loop(self):
        if dist.get_rank() == 0:
            print("***** Running training *****")
            logger.info(f"  Max steps = {self.total_steps}")
            logger.info(f"  Instantaneous batch size per GPU = {self.batch_size}")
            bs = self.batch_size * self.gradient_accumulation_steps * (dist.get_world_size())
            logger.info(
                f"  Total train batch size (w. parallel, distributed & accumulation) = {bs}"
            )
            logger.info(f"  Total warm up steps = {self.warmup_steps}")
            logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
            
        if self.config.continue_train and (
            abs(self.optimizer.param_groups[0]['lr'] - self.scheduler.get_lr()[0]) > 1e-10):
            self.scheduler.step()

        while self.global_step < self.total_steps:            
            epoch_iterator = tqdm(
                self.data, desc="Iteration", disable=dist.get_rank() not in [-1, 0]
            )

            self.model.zero_grad()
            self.model.train()
            for step, batch in enumerate(epoch_iterator):
                # step_ratio = 1.0
                # if self.config.use_step_ratio:
                #     step_ratio = float(self.global_step / self.total_steps)
                # self.forward_backward(batch, step_ratio)

                self.forward_backward(batch)

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    
                    if self.gradient_clipping > 0:
                        if self.config.use_AMP:
                            self.scaler.unscale_(self.optimizer)
                        self.grad_clip()
                    
                    if self.config.use_AMP:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.model.zero_grad()

                if self.global_step % self.log_interval == 0:
                    self.writer.add_scalar(
                        tag='learning_rate', 
                        scalar_value=self.optimizer.param_groups[0]['lr'], 
                        global_step=self.global_step
                    )
                    
                # ema schedule. It actually save a shadow model for evalution.
                # It doesn't change the training process.
                for rate, params in zip(self.ema_rate, self.ema_params):
                    self.update_ema(params, self.master_params, rate=rate)
                self.global_step += 1
                
                # dev dataset for evaluation
                if self.dev_data is not None and self.global_step % self.eval_interval == 0:
                    if dist.get_rank() == 0:
                        print('eval on validation set...')
                        for step, batch in tqdm(enumerate(self.dev_data)):
                            if step > 50:
                                break
                            # self.forward_only(step, batch, step_ratio)
                            self.forward_only(step, batch)

                # save model
                if (self.total_steps - self.global_step) < 30000 and self.global_step % self.save_interval == 0:
                    self.save()
                elif self.global_step % 10000 == 0:
                    self.save()

    # def forward_backward(self, batch, step_ratio):
    def forward_backward(self, batch):

        if self.train_type == 's2s':
            # the timestep t is random sample.
            if self.config.schedule_sampler == 'uniform':
                t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], 
                                                          self.device, 
                                                        #   step_ratio=1.0-step_ratio,
                                                          )
            else:
                t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], 
                                                          self.device, 
                                                          seq_len=batch['length'].to(self.device),
                                                        #   step_ratio=1.0-step_ratio,
                                                          )
            if self.config.use_AMP:
                with autocast(device_type='cuda', dtype=torch.float16):
                    losses = self.diffusion.training_losses(self.model, batch, t)
                    
                    # loss moment
                    if self.config.schedule_sampler == 'loss-second-moment':
                        self.schedule_sampler.update_with_local_losses(
                            t, losses["loss"].detach()
                        )
                        
                    if self.config.loss_aware:
                        self.schedule_sampler.update_with_local_losses(
                            t, losses["loss"].detach()
                        )
                    
                    if self.config.pred_len:
                        loss = (losses["loss"] * weights * batch['tgt_attention_mask'].to(self.device)).mean()
                    else:
                        loss = (losses["loss"] * weights).mean()
                    loss = loss / self.gradient_accumulation_steps
                
                # if self.config.grad_penalty:
                #     scaled_grad_params = torch.autograd.grad(
                #         outputs=self.scaler.scale(loss), inputs=self.model.parameters(), create_graph=True)

                #     inv_scale = 1. / self.scaler.get_scale()
                #     grad_params = [p * inv_scale for p in scaled_grad_params]

                #     with autocast(device_type='cuda', dtype=torch.float16):
                #         grad_norm = 0
                #         for grad in grad_params:
                #             grad_norm += grad.pow(2).sum()
                #         grad_norm = grad_norm.sqrt()
                #         loss = loss + grad_norm    
                
            else:
                losses = self.diffusion.training_losses(self.model, batch, t)
                
                # loss moment
                if self.config.schedule_sampler == 'loss-second-moment':
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )
                
                if self.config.loss_aware:
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()
                loss = loss / self.gradient_accumulation_steps
                
                # if self.config.grad_penalty:
                #     grad_params = torch.autograd.grad(
                #         outputs=loss, inputs=self.model.parameters(), create_graph=True)
                #     grad_norm = 0
                #     for grad in grad_params:
                #         grad_norm += grad.pow(2).sum()
                #     grad_norm = grad_norm.sqrt()
                #     loss = loss + grad_norm

        else:
            return NotImplementedError
        
        if self.config.use_AMP:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.global_step % self.log_interval == 0:
            for key, value in losses.items():
                if self.config.pred_len:
                    losses[key] = (value * weights * batch['tgt_attention_mask'].to(self.device)).mean()
                else:
                    losses[key] = (value * weights).mean()
                self.writer.add_scalar(
                    tag=f'train_loss/{key}', scalar_value=losses[key], global_step=self.global_step)

        # for index, (name, p) in enumerate(self.model.module.named_parameters()):
        #     if p.grad == None:
        #         print(index, name)

    # def forward_only(self, step, batch, step_ratio):
    def forward_only(self, step, batch):
        with torch.no_grad():
            self.model.zero_grad()
            if self.train_type == 's2s':
                if self.config.schedule_sampler == 'uniform':
                    t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], 
                                                              self.device, 
                                                            #   step_ratio=1.0-step_ratio
                                                            )
                else:
                    t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], 
                                                              self.device, 
                                                              seq_len=batch['length'].to(self.device),
                                                            #   step_ratio=1.0-step_ratio,
                                                            )
                if self.config.use_AMP:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        losses = self.diffusion.training_losses(self.model, batch, t, is_dev=True)
                else:
                    losses = self.diffusion.training_losses(self.model, batch, t, is_dev=True)
            else:
                return NotImplementedError

            for key, value in losses.items():
                if 'acc' not in key:
                    if self.config.pred_len:
                        value = (value * weights * batch['tgt_attention_mask'].to(self.device)).mean()
                    else:
                        value = (value * weights).mean()
                self.writer.add_scalar(
                    tag=f'eval_loss/{key}', scalar_value=value, global_step=self.global_step)

    def check_load(self):
        model_checkpoint_files = []
        ema_checkpoint_files = []
        for item in os.scandir(self.model_path):
            if item.is_file():
                if "model_checkpoint" in item.path:
                    model_checkpoint_files.append(item.path)
                if "ema" in item.path:
                    ema_checkpoint_files.append(item.path)

        if not self.config.load_from_ema and len(model_checkpoint_files) != 0:
            model_checkpoint_files.sort(key=lambda f: int(
                f.split('model_checkpoint-')[1]), reverse=True)
            if dist.get_rank() == 0:
                logger.info("***** load " + model_checkpoint_files[0] + " *****")

            model_saved_state = load_states_from_checkpoint(
                model_checkpoint_files[0], dist.get_rank())
            self.global_step = self._load_saved_state(model_saved_state)

        elif self.config.load_from_ema and len(ema_checkpoint_files) != 0:
            ema_checkpoint_files.sort(key=lambda f: int(
                f.split('checkpoint-')[-1]), reverse=True)
            if dist.get_rank() == 0:
                logger.info("***** load " + ema_checkpoint_files[0] + " *****")
            
            ema_saved_state = load_states_from_checkpoint(
                ema_checkpoint_files[0], dist.get_rank())
            self.ema_params = [
                [ema_saved_state.model_dict[name].to(self.device) 
                 for name, _ in self.model.named_parameters()]
                for _ in range(len(self.ema_rate))
            ]
            self.global_step = self._load_saved_state(ema_saved_state)

        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]
            if dist.get_rank() == 0:
                logger.info("***** there are no checkpoint in " + self.model_path + " *****")
                
    def _load_saved_state(self, saved_state: CheckpointState):
        self.global_step = saved_state.offset
        if dist.get_rank() == 0:
            logger.info('Loading checkpoint @ step=%s', self.global_step)
            print('Loading saved model state ...')
        if self.config.continue_train:
            saved_state.scheduler_dict['base_lrs'] = [self.config.lr]
            if dist.get_rank() == 0:
                logger.info('Now the learning rate is %s', saved_state.scheduler_dict['base_lrs'])
        # set strict=False if you use extra projection
        self.model.load_state_dict(saved_state.model_dict)
        self.optimizer.load_state_dict(saved_state.optimizer_dict)
        self.scheduler.load_state_dict(saved_state.scheduler_dict)
        self.master_params = list(self.model.parameters())
        return self.global_step

    def save(self):
        def save_checkpoint(rate, ema_params):
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
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
            state = CheckpointState(
                model_state_dict, opt_state_dict, sch_state_dict, offset,)
            if not rate:
                ckpt_path = os.path.join(
                    self.model_path, 'model_checkpoint-' + str(offset))
            else:
                ckpt_path = os.path.join(
                    self.model_path, 'ema_' + str(rate) + '_checkpoint-' + str(offset))

            torch.save(state._asdict(), ckpt_path)
            if dist.get_rank() == 0:
                logger.info('Saved checkpoint at %s', ckpt_path)

        if dist.get_rank() == 0:
            save_checkpoint(0, None)
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params)

    def grad_clip(self):
        max_grad_norm = self.gradient_clipping  # 3.0
        if hasattr(self.optimizer, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.optimizer.clip_grad_norm(max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),  # amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )

    def update_ema(self, target_params, source_params, rate=0.99):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.

        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)
