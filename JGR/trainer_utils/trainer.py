# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
from email import generator
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
from tkinter import E
import warnings
from logging import StreamHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple

from tqdm.auto import tqdm


# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)
import json
import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    IntervalStrategy,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)

from .utils import (
    TrainOutput,
    EvalLoopOutput_ours,
    PredictionOutput_ours
)
from .training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers.utils.modeling_auto_mapping import MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES


__version__ = "4.8.1"

_is_torch_generator_available = False
_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# if is_apex_available():
#     from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
else:
    import torch.distributed as dist

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

if is_training_run_on_sagemaker():
    logging.add_handler(StreamHandler(sys.stdout))


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)

def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0




class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.
    Args:
        model (:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.
            .. note::
                :class:`~transformers.Trainer` is optimized to work with the :class:`~transformers.PreTrainedModel`
                provided by the library. You can still use your own models defined as :obj:`torch.nn.Module` as long as
                they work the same way as the 🤗 Transformers models.
        args (:class:`~transformers.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~transformers.TrainingArguments` with the ``output_dir`` set to a directory named `tmp_trainer` in
            the current directory if not provided.
        data_collator (:obj:`DataCollator`, `optional`):
            The function to use to form a batch from a list of elements of :obj:`train_dataset` or :obj:`eval_dataset`.
            Will default to :func:`~transformers.default_data_collator` if no ``tokenizer`` is provided, an instance of
            :func:`~transformers.DataCollatorWithPadding` otherwise.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset` or :obj:`torch.utils.data.dataset.IterableDataset`, `optional`):
            The dataset to use for training. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.
            Note that if it's a :obj:`torch.utils.data.dataset.IterableDataset` with some randomization and you are
            training in a distributed fashion, your iterable dataset should either use a internal attribute
            :obj:`generator` that is a :obj:`torch.Generator` for the randomization that must be identical on all
            processes (and the Trainer will manually set the seed of this :obj:`generator` at each epoch) or have a
            :obj:`set_epoch()` method that internally sets the seed of the RNGs used.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
             ``model.forward()`` method are automatically removed.
        tokenizer (:class:`PreTrainedTokenizerBase`, `optional`):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        callbacks (List of :obj:`~transformers.TrainerCallback`, `optional`):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in :doc:`here <callback>`.
            If you want to remove one of the default callbacks used, use the :meth:`Trainer.remove_callback` method.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.
    Important attributes:
        - **model** -- Always points to the core model. If using a transformers model, it will be a
          :class:`~transformers.PreTrainedModel` subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under ``DeepSpeed``,
          the inner model is wrapped in ``DeepSpeed`` and then again in ``torch.nn.DistributedDataParallel``. If the
          inner model hasn't been wrapped, then ``self.model_wrapped`` is the same as ``self.model``.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to :obj:`False` if model parallel or deepspeed is used, or if the default
          ``TrainingArguments.place_model_on_device`` is overridden to return :obj:`False` .
        - **is_in_train** -- Whether or not a model is currently running ``train`` (e.g. when ``evaluate`` is called
          while in ``train``)
    """

    from transformers.trainer_pt_utils import log_metrics, metrics_format, save_metrics, save_state
    from .trainer_funcs import (
        get_train_dataloader, get_eval_dataloader, get_test_dataloader, _get_train_sampler, _get_eval_sampler,
        create_optimizer, create_scheduler, _get_learning_rate,
        num_examples, _prepare_inputs,
        _save_checkpoint, _wrap_model, _load_state_dict_in_model, save_model, _rotate_checkpoints, _sorted_checkpoints,
        is_world_process_zero, is_local_process_zero,
        floating_point_ops, store_flos,
        _nested_gather, _nested_gather_object,
        )

    def __init__(
        self,
        generator_model: Union[PreTrainedModel, nn.Module] = None,
        reranker_model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        data_collator_eval: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        generator_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        reranker_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.Optimizer] = (None, None),
        lr_schedulers: Tuple[torch.optim.lr_scheduler.LambdaLR, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        self.is_in_train = False

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        if (hasattr(generator_model, "is_parallelizable") and generator_model.is_parallelizable and generator_model.model_parallel
            and hasattr(reranker_model, "is_parallelizable") and reranker_model.is_parallelizable and reranker_model.model_parallel):
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False


        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full fp16 eval - since the model needs to be half'ed first
        # 4. Sharded DDP - same as MP
        self.place_model_on_device = args.place_model_on_device
        if (
            self.is_model_parallel
            or (args.fp16_full_eval and not args.do_train)
        ):
            self.place_model_on_device = False

        default_collator = default_data_collator if generator_tokenizer is None else DataCollatorWithPadding(generator_tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.data_collator_eval = data_collator_eval if data_collator_eval is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.generator_tokenizer = generator_tokenizer
        self.reranker_tokenizer = reranker_tokenizer

        if self.place_model_on_device:
            generator_model = generator_model.to(args.device)
            reranker_model = reranker_model.to(args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.generator_model_wrapped = generator_model
        self.generator_model = generator_model
        self.reranker_model_wrapped = reranker_model
        self.reranker_model = reranker_model

        self.compute_metrics = compute_metrics
        self.generator_optimizer, self.reranker_optimizer = optimizers
        self.generator_scheduler, self.reranker_scheduler = lr_schedulers
        

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create clone of distant repo and output directory if needed
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not isinstance(train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")

        self._signature_columns = None

        # Mixed precision setup
        self.use_apex = False
        self.use_amp = False
        self.fp16_backend = None

        if args.fp16:
            if args.fp16_backend == "auto":
                self.fp16_backend = "amp" if _is_native_amp_available else "apex"
            else:
                self.fp16_backend = args.fp16_backend
            logger.info(f"Using {self.fp16_backend} fp16 backend")

        if args.fp16:  # deepspeed manages its own fp16
            if self.fp16_backend == "amp":
                self.use_amp = True
                if is_sagemaker_mp_enabled():
                    self.scaler = smp.amp.GradScaler()
                else:
                    self.scaler = torch.cuda.amp.GradScaler()
            else:
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True


        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.total_flos = 0
        self.current_flos = 0
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        self.training_bar = None
        self.prediction_bar = None
        
        self.label_names = ["labels"]

        # initialize a state tracker
        self.metrics_tracker = {}
        if self.args.do_train:
            self.metrics_tracker['train_metrics'] = []
        if self.args.do_eval:
            self.metrics_tracker['eval_metrics'] = []

        self.reward_tracker = {
            'metric_rewards': [],
            'reranker_rewards': []
        }

        # very last
        self._memory_tracker.stop_and_update_metrics()


    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Main training entry point.
        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self.generator_model = self.generator_model.to(args.device)
            self.reranker_model = self.reranker_model.to(args.device)

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            num_train_epochs = int(args.num_train_epochs)
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size


        self.generator_optimizer = self.create_optimizer(self.generator_optimizer, self.generator_model, self.args.generator_learning_rate)
        self.reranker_optimizer = self.create_optimizer(self.reranker_optimizer, self.reranker_model, self.args.reranker_learning_rate)
        self.generator_scheduler = self.create_scheduler(self.generator_scheduler, self.generator_optimizer, max_steps)
        self.reranker_scheduler = self.create_scheduler(self.reranker_scheduler, self.reranker_optimizer, max_steps)

        generator_model = self._wrap_model(self.generator_model_wrapped)
        reranker_model = self._wrap_model(self.reranker_model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if generator_model is not self.generator_model:
            self.generator_model_wrapped = generator_model
        if reranker_model is not self.reranker_model:
            self.reranker_model_wrapped = reranker_model

        # Check if saved optimizer or scheduler states exist
        # self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        
        start_time = time.time()
        self.best_metric = None
        self.best_model_checkpoint = None
        self.should_training_stop = False
        epochs_trained = 0
        self.global_step = 0
        self._globalstep_last_logged = 0
        self.current_step = 0
        train_flag = 0 # train reranker
        steps_trained_in_current_epoch = 0
        if self.is_local_process_zero() and not self.args.disable_tqdm:
            # setup tqdm bar
            self.training_bar = tqdm(total=max_steps)
        steps_trained_progress_bar = None

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        reranker_tr_loss = torch.tensor(0.0).to(args.device)
        generator_tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._reranker_total_loss_scalar = 0.0
        self._generator_total_loss_scalar = 0.0
        generator_model.zero_grad()
        reranker_model.zero_grad()


        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )

            for step, inputs in enumerate(epoch_iterator):
                if train_flag == 0:
                    # train reranker
                    
                    # Skip past any already trained steps if resuming training
                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with generator_model.no_sync(), reranker_model.no_sync():
                            reranker_tr_loss += self.training_step_reranker(generator_model, reranker_model, inputs)
                    else:
                        reranker_tr_loss += self.training_step_reranker(generator_model, reranker_model, inputs)
                    self.current_flos += float(self.floating_point_ops(inputs))

                    if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                    ):
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if self.use_amp:
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.reranker_optimizer)

                            if hasattr(self.reranker_optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.reranker_optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(reranker_model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                reranker_model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.reranker_optimizer) if self.use_apex else reranker_model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        if is_torch_tpu_available():
                            xm.optimizer_step(self.reranker_optimizer)
                        elif self.use_amp:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.reranker_optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                        else:
                            self.reranker_optimizer.step()
                            self.reranker_scheduler.step()

                        reranker_model.zero_grad()
                        generator_model.zero_grad()
                        self.global_step += 1
                        epoch = epoch + (step + 1) / steps_in_epoch

                        # log eval save
                        self._handle_log_save_eval_on_step([generator_model, reranker_model], [generator_tr_loss, reranker_tr_loss], train_flag)

                        if self.is_local_process_zero() and not self.args.disable_tqdm:
                            self.training_bar.update(self.global_step - self.current_step)
                            self.current_step = self.global_step

                        if self.global_step >= max_steps:
                            self.should_training_stop = True
                            break
                        # we need to swith
                        if self.global_step % self.args.iteration_steps > self.args.iteration_reranker_steps:
                            # train generatro
                            train_flag = 1
                        elif 0 < self.global_step % self.args.iteration_steps < self.args.iteration_reranker_steps:
                            # train reranker
                            train_flag = 0
                else:
                    # we need to implement training for generator here
                    # Skip past any already trained steps if resuming training
                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with generator_model.no_sync(), reranker_model.no_sync():
                            generator_tr_loss += self.training_step_generator(generator_model, reranker_model, inputs)
                    else:
                        generator_tr_loss += self.training_step_generator(generator_model, reranker_model, inputs)
                    self.current_flos += float(self.floating_point_ops(inputs))

                    if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                    ):
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if self.use_amp:
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.generator_optimizer)

                            if hasattr(self.generator_optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.generator_optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(generator_model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                generator_model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.generator_optimizer) if self.use_apex else generator_model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        if is_torch_tpu_available():
                            xm.optimizer_step(self.generator_optimizer)
                        elif self.use_amp:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.generator_optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                        else:
                            self.generator_optimizer.step()
                            self.generator_scheduler.step()

                        # loss.backward()

                        reranker_model.zero_grad()
                        generator_model.zero_grad()
                        self.global_step += 1
                        epoch = epoch + (step + 1) / steps_in_epoch

                        # log eval save
                        self._handle_log_save_eval_on_step([generator_model, reranker_model], [generator_tr_loss, reranker_tr_loss], train_flag)

                        if self.is_local_process_zero() and not self.args.disable_tqdm:
                            self.training_bar.update(self.global_step - self.current_step)
                            self.current_step = self.global_step

                        if self.global_step >= max_steps:
                            self.should_training_stop = True
                            break

                        # we need to swith
                        if self.global_step % self.args.iteration_steps > self.args.iteration_reranker_steps:
                            # train generatro
                            train_flag = 1
                        elif 0 < self.global_step % self.args.iteration_steps < self.args.iteration_reranker_steps:
                            # train reranker
                            train_flag = 0


            self._handle_log_save_eval_on_epoch([generator_model, reranker_model], [generator_tr_loss, reranker_tr_loss], train_flag)

            if self.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.best_model_checkpoint} (score: {self.best_metric})."
            )
            # We load the model state dict on the CPU to avoid an OOM error.
            # state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME), map_location="cpu")
            generator_state_dict = torch.load(os.path.join(self.best_model_checkpoint, "generator", WEIGHTS_NAME), map_location="cpu")
            reranker_state_dict = torch.load(os.path.join(self.best_model_checkpoint, "reranker", WEIGHTS_NAME), map_location="cpu")
            # If the model is on the GPU, it still works!
            self._load_state_dict_in_model(generator_state_dict, reranker_state_dict)


        # add remaining tr_loss
        self._generator_total_loss_scalar += generator_tr_loss.item()
        self._reranker_total_loss_scalar += reranker_tr_loss.item()
        
        generator_train_loss = self._generator_total_loss_scalar / self.global_step
        reranker_train_loss = self._reranker_total_loss_scalar / self.global_step
        # train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=max_steps)
        self.store_flos()
        metrics["total_flos"] = self.total_flos
        metrics["generator_train_loss"] = generator_train_loss
        metrics["reranker_train_loss"] = reranker_train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)


        if self.is_local_process_zero() and not self.args.disable_tqdm:
            self.training_bar.write(str(metrics))
            self.training_bar.close()
            self.training_bar = None

        return TrainOutput(self.global_step, generator_train_loss, reranker_train_loss, metrics)


    def _handle_log_save_eval_on_step(self,  models, tr_losses, train_flag):
        # log
        if (
                self.args.logging_strategy == IntervalStrategy.STEPS
                and self.args.logging_steps > 0
                and self.global_step % self.args.logging_steps == 0
            ):
            generator_tr_loss, reranker_tr_loss = tr_losses
            logs: Dict[str, float] = {}
            generator_tr_loss_scalar = generator_tr_loss.item()
            reranker_tr_loss_scalar = reranker_tr_loss.item()
            # reset tr_loss to zero
            generator_tr_loss -= generator_tr_loss
            reranker_tr_loss -= reranker_tr_loss
            logs['steps'] = self.global_step
            
            if train_flag == 0:
                logs["reranker_loss"] = round(reranker_tr_loss_scalar / (self.global_step - self._globalstep_last_logged), 4)
                logs["learning_rate"] = self._get_learning_rate(self.reranker_scheduler)
            else:
                logs["generator_loss"] = round(generator_tr_loss_scalar / (self.global_step - self._globalstep_last_logged), 4)
                logs["learning_rate"] = self._get_learning_rate(self.generator_scheduler)

            self._reranker_total_loss_scalar += reranker_tr_loss_scalar
            self._generator_total_loss_scalar += generator_tr_loss_scalar
            self._globalstep_last_logged = self.global_step
            if self.is_local_process_zero():
                self.metrics_tracker['train_metrics'].append(logs)
                # for key, value in logs.items():
                #     tb_writer.add_scalar(key, value, global_step)
                print(logs)
                # self.training_bar.write(str(logs))
                logger.info(json.dumps({**logs, **{"step": self.global_step}}))

        # evaluate
        metrics = None
        should_save = False
        if self.args.evaluation_strategy == IntervalStrategy.STEPS and self.global_step % self.args.eval_steps == 0:
            if self.args.load_best_model_at_end:
                should_save = True
            metrics = self.evaluate()
            if self.is_local_process_zero():
                self.metrics_tracker['eval_metrics'].append(metrics)

        # save, if we set load_best_model_at_end, then the saving strategy is negelected, the models will be saved when at each evaluation step
        if (
            not self.args.load_best_model_at_end
            and self.args.save_strategy == IntervalStrategy.STEPS
            and self.args.save_steps > 0
            and self.global_step % self.args.save_steps == 0
            ) or should_save:
            self._save_checkpoint(metrics=metrics)

    def _handle_log_save_eval_on_epoch(self,  models, tr_losses, train_flag):
        # log
        if self.args.logging_strategy == IntervalStrategy.EPOCH:
            generator_tr_loss, reranker_tr_loss = tr_losses
            logs: Dict[str, float] = {}
            generator_tr_loss_scalar = generator_tr_loss.item()
            reranker_tr_loss_scalar = reranker_tr_loss.item()
            # reset tr_loss to zero
            generator_tr_loss -= generator_tr_loss
            reranker_tr_loss -= reranker_tr_loss
            logs['steps'] = self.global_step
            if train_flag == 0:
                logs["reranker_loss"] = round(reranker_tr_loss_scalar / (self.global_step - self._globalstep_last_logged), 4)
                logs["learning_rate"] = self._get_learning_rate(self.reranker_scheduler)
            else:
                logs["generator_loss"] = round(generator_tr_loss_scalar / (self.global_step - self._globalstep_last_logged), 4)
                logs["learning_rate"] = self._get_learning_rate(self.generator_scheduler)

            self._reranker_total_loss_scalar += reranker_tr_loss_scalar
            self._generator_total_loss_scalar += generator_tr_loss_scalar
            self._globalstep_last_logged = self.global_step
            if self.is_local_process_zero():
                self.metrics_tracker['train_metrics'].append(logs)
                # for key, value in logs.items():
                #     tb_writer.add_scalar(key, value, global_step)
                print(logs)
                # self.training_bar.write(str(logs))
                logger.info(json.dumps({**logs, **{"step": self.global_step}}))

        # evaluate
        metrics = None
        should_save = False
        if self.args.evaluation_strategy == IntervalStrategy.EPOCH:
            if self.args.load_best_model_at_end:
                should_save = True
            metrics = self.evaluate()
            if self.is_local_process_zero():
                self.metrics_tracker['eval_metrics'].append(metrics)

        # save, if we set load_best_model_at_end, then the saving strategy is negelected, the models will be saved when at each evaluation step
        if self.args.save_strategy == IntervalStrategy.EPOCH or should_save:
            self._save_checkpoint(metrics=metrics)


    def training_step_reranker(self, generator_model: nn.Module, reranker_model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        generator_model.eval()
        reranker_model.train()

        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss_reranker(generator_model, reranker_model, inputs)
        else:
            loss = self.compute_loss_reranker(generator_model, reranker_model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 :
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        return loss.detach()

    
    def training_step_generator(self, generator_model: nn.Module, reranker_model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        generator_model.train()
        reranker_model.eval()

        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss_generator(generator_model, reranker_model, inputs)
        else:
            loss = self.compute_loss_generator(generator_model, reranker_model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 :
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()


        return loss.detach()

    def compute_loss_reranker_eval(self, reranker_model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = reranker_model(input_ids = inputs['reranker_input_ids'], attention_mask=inputs['reranker_attention_mask'])
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def compute_loss_reranker(self, generator_model, reranker_model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        target_text = inputs.pop("target_text")
        source_text = inputs.pop('source_text')
        
        # use sampling not beam search
        gen_kwargs = {
            "max_length": self.args.generator_max_target_length,
            "num_beams": 1,
            "do_sample":True,
            "num_return_sequences": self.args.num_cand_generated,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False
        }
        generated_tokens = unwrap_model(generator_model).generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        ) # (B*C, max_target_len)
        generated_tokens = generated_tokens.detach()
        generated_seqs = self.generator_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

        _, candidate_texts, candidate_scores = self.compute_metrics.get_candidates(target_text,  generated_seqs, 
                                self.args.num_cand_generated, self.args.num_cand_picked, self.args.candidate_pick_strategy) # (B* (C-1)), list with length of B*C
        
        # self.reward_tracker['scores'].append(torch.var(candidate_scores, dim=1).mean().item())
        
        # process input for input
        reranker_input_ids, reranker_attention_mask = self._get_reranker_input_ids(source_text, candidate_texts) # both (B, C, L)

        reranker_output = reranker_model(input_ids=reranker_input_ids, attention_mask = reranker_attention_mask)

        # self.reward_tracker['reranker_logits'].append(torch.var(reranker_output.logits, dim=1).mean().item())
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = reranker_output[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(reranker_output, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = reranker_output["loss"] if isinstance(reranker_output, dict) else reranker_output[0]

        return (loss, reranker_output) if return_outputs else loss


    def compute_loss_generator(self, generator_model, reranker_model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        target_text = inputs.pop("target_text")
        source_text = inputs.pop('source_text')

        # use sampling not beam search
        gen_kwargs = {
            "max_length": self.args.generator_max_target_length,
            "num_beams": 1,
            "do_sample":True,
            "num_return_sequences": self.args.generator_num_cand_generated,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "return_dict_in_generate": True
        }
        generator_outputs = unwrap_model(generator_model).generate_for_training(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        ) # (B*num_candidates, L)
        generated_tokens = generator_outputs.sequences # (B*num_candidates, L+1)
        generated_logits = generator_outputs.scores # (B*num_candidates, L, V) without softmax in -1 dimension
        generated_lens = generator_outputs.seq_lens # (B*num_candidates, )
        generated_seqs = self.generator_tokenizer.batch_decode(
                    generated_tokens.detach(), skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

        # candidate_indices (B *C), candidate_texts list with lenght of B*C, candidate_scores (B, C)
        candidate_indices, candidate_texts, candidate_scores = self.compute_metrics.get_candidates(target_text, generated_seqs, 
                                self.args.generator_num_cand_generated, self.args.generator_num_cand_picked,  self.args.candidate_pick_strategy) # (B* C), list with length of B*C
        
        candidate_indices = candidate_indices.to(self.args.device)
        candidate_scores = candidate_scores.to(self.args.device)

        eps = 1e-7
        # get the probability of generated tokens
        seq_len = generated_logits.size(1)
        vocab_size = generated_logits.size(-1)
        generated_probs = nn.functional.softmax(generated_logits, dim=-1)
        generated_probs = generated_probs.contiguous().view(-1, vocab_size) # (B*num_candidates*L, V)
        generated_tokens_indices = generated_tokens[:,1:].contiguous().view(-1).unsqueeze(1)
        generated_probs = torch.gather(generated_probs, 1, generated_tokens_indices)
        generated_probs = generated_probs.view(-1, seq_len) # (B * num_candates, L )
        
        

        generated_probs = torch.index_select(generated_probs, 0, candidate_indices) # (B * C, L) 

        # if self.args.reward_type == 'reranker_softmax':
        #     # we take reshaping here to avoid no gradient
        #     generated_probs = generated_probs.view(-1, self.args.generator_num_cand_picked, seq_len)
        #     generated_probs = generated_probs[:,1:,:]
        #     generated_probs = generated_probs.contiguous().view(-1, seq_len)


        # compute reward
        with torch.no_grad():
            if self.args.use_baseline_reward:
                greedy_gen_kwargs = {
                    "max_length": self.args.generator_max_target_length,
                    "num_beams": 1,
                    "do_sample":False,
                    "num_return_sequences": 1,
                    "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
                    "return_dict_in_generate": False
                }
                greedy_outputs = unwrap_model(generator_model).generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **greedy_gen_kwargs,
                ) # (B*num_candidates, L)
                greedy_texts = self.generator_tokenizer.batch_decode(
                    greedy_outputs.detach(), skip_special_tokens=True, clean_up_tokenization_spaces=True
                ) # list with length of B


            reranker_input_ids, reranker_attention_mask = self._get_reranker_input_ids(source_text, candidate_texts) # both (B, C, L)

            reranker_output = reranker_model(input_ids=reranker_input_ids, attention_mask = reranker_attention_mask)

            reranker_logits = reranker_output.logits #(B, C)

            # self.reward_tracker['scores'].append(torch.var(candidate_scores, dim=1).mean().item())

            # self.reward_tracker['reranker_logits'].append(torch.var(reranker_output.logits, dim=1).mean().item())

            # if self.args.reranker_loss_type == 'binary':
            #     rewards_probs = torch.sigmoid(reranker_logits) #(B, C-1)
            # else:
            #     rewards_probs = torch.softmax(reranker_logits, dim = 1)
            
            # rewards = torch.log(rewards_probs+eps)
            # rewards_max = torch.max(rewards, dim=1, keepdim =True).values # (B, 1)
            # rewards_min = torch.min(rewards, dim=1, keepdim =True).values # (B, 1)
            # rewards_base = (rewards_max+rewards_min) / 2
            # rewards_base = rewards[:,:1] # (B, 1)
            reranker_rewards_baseline = torch.mean(reranker_logits, dim = -1, keepdim = True) #(B, 1)
            metric_rewards_baseline = torch.mean(candidate_scores, dim = -1, keepdim= True)

            reranker_rewards = reranker_logits - reranker_rewards_baseline
            metric_rewards = candidate_scores - metric_rewards_baseline

            self.reward_tracker['reranker_rewards'].append(reranker_rewards.detach().cpu().numpy().tolist())
            self.reward_tracker['metric_rewards'].append(metric_rewards.detach().cpu().numpy().tolist())

            if self.args.normalize_rewards:
                rererank_rewards_std = torch.std(reranker_rewards, dim=1, keepdim = True)
                metric_rewards_std = torch.std(metric_rewards, dim=1, keepdim = True)
                reranker_rewards = reranker_rewards / (rererank_rewards_std + eps)
                metric_rewards = metric_rewards / (metric_rewards_std + eps)
            # rewards = torch.relu(rewards_base - rewards) #always positive
            rewards = self.args.reranker_reward_scaler * reranker_rewards + self.args.metric_reward_scaler * metric_rewards
            rewards = rewards.view(-1) #(B* C)
            rewards = rewards.unsqueeze(1).expand_as(generated_probs)

        
        rl_loss = -rewards*torch.log(generated_probs+eps) # (B*C-1, L)
        rl_loss = rl_loss.mean()
        loss = rl_loss


        # supervise training
        if self.args.generator_supervised:
            generator_sup_outputs = generator_model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'], labels=inputs['labels'])
            sup_loss = generator_sup_outputs["loss"] if isinstance(generator_sup_outputs, dict) else generator_sup_outputs[0]
            loss += self.args.generator_supervised_lambda * sup_loss


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = reranker_output[self.args.past_index]

        return (loss, generator_outputs) if return_outputs else loss




    def _get_reranker_input_ids(self, source_texts: List[str], candidates: List[str]):
        """
            source_texts: list of source with length == B
            candidates: list of candidates with length == B*C
        """
        def encode_seq(seq):
            """
            encode the tokenized or untokenized sequence to token ids
            """
            if seq == [] or seq == "":
                return []
            else:
                return self.reranker_tokenizer.encode(seq, add_special_tokens=False)


        num_cand = len(candidates) // len(source_texts)
        reranker_input_ids = []
        for i,s in enumerate(source_texts):
            source_ids = encode_seq(s)[:self.args.reranker_max_source_length]
            cs = candidates[i*num_cand: (i+1)*num_cand]
            for c in cs:
                input_id = [self.reranker_tokenizer.cls_token_id] + source_ids
                input_id += [self.reranker_tokenizer.sep_token_id]
                input_id += encode_seq(c)[:self.args.reranker_max_target_length]
                reranker_input_ids.append(torch.LongTensor(input_id))
        
        reranker_input_ids = pad_sequence(reranker_input_ids, batch_first = True, padding_value = self.reranker_tokenizer.pad_token_id)
        reranker_input_ids = reranker_input_ids.view(len(source_texts), num_cand, -1)
        attention_mask = reranker_input_ids != self.reranker_tokenizer.pad_token_id

        reranker_input_ids = reranker_input_ids.to(self.args.device)
        attention_mask = attention_mask.to(self.args.device)

        return reranker_input_ids, attention_mask
        
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop =  self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        if self.is_local_process_zero():
            print(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        if self.is_local_process_zero() and not self.args.disable_tqdm:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput_ours:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.
        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)
        .. note::
            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.
        Returns: `NamedTuple` A namedtuple with the following keys:
            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        if self.is_local_process_zero() and not self.args.disable_tqdm:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput_ours(generator_predictions=output.generator_predictions, reranker_predictions=output.reranker_predictions, label_ids=output.label_ids, metrics=output.metrics)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput_ours:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        reranker_model = self._wrap_model(self.reranker_model, training=False)
        generator_model = self._wrap_model(self.generator_model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            reranker_model = reranker_model.half().to(self.args.device)
            generator_model = generator_model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        reranker_model.eval()

        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        generator_losses_host = None
        reranker_losses_host = None
        generator_preds_host = None
        reranker_preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        generator_all_losses = None
        reranker_all_losses = None
        generator_all_preds = None
        reranker_all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            generator_loss, reranker_loss, generator_logits, reranker_logits, labels = self.prediction_step(generator_model, reranker_model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            # loss: loss, preds: list of selected sequences, targets: the ground truth sequences

            # Update containers on host
            if generator_loss is not None:
                generator_losses = self._nested_gather(generator_loss.repeat(batch_size))
                generator_losses_host = generator_losses if generator_losses_host is None else torch.cat((generator_losses_host, generator_losses), dim=0)
            if reranker_loss is not None:
                reranker_losses = self._nested_gather(reranker_loss.repeat(batch_size))
                reranker_losses_host = reranker_losses if reranker_losses_host is None else torch.cat((reranker_losses_host, reranker_losses), dim=0)
            if generator_logits is not None:
                # logits = self._pad_across_processes(logits)
                generator_logits = self._nested_gather_object(generator_logits)
                generator_preds_host = generator_logits if generator_preds_host is None else generator_preds_host + generator_logits
            if reranker_logits is not None:
                # logits = self._pad_across_processes(logits)
                reranker_logits = self._nested_gather_object(reranker_logits)
                reranker_preds_host = reranker_logits if reranker_preds_host is None else reranker_preds_host + reranker_logits
            if labels is not None:
                # labels = self._pad_across_processes(labels)
                labels = self._nested_gather_object(labels)
                labels_host = labels if labels_host is None else labels_host + labels


            # setup tqdm bar
            if self.is_local_process_zero() and isinstance(dataloader.dataset, collections.abc.Sized) and not self.args.disable_tqdm:
                if self.prediction_bar is None:
                    self.prediction_bar = tqdm(total=len(dataloader), leave=self.training_bar is None)
                self.prediction_bar.update(1)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if generator_losses_host is not None:
                    generator_losses = nested_numpify(generator_losses_host)
                    generator_all_losses = generator_losses if generator_all_losses is None else np.concatenate((generator_all_losses, generator_losses), axis=0)
                if reranker_losses_host is not None:
                    reranker_losses = nested_numpify(reranker_losses_host)
                    reranker_all_losses = reranker_losses if reranker_all_losses is None else np.concatenate((reranker_all_losses, reranker_losses), axis=0)
                if generator_preds_host is not None:
                    generator_all_preds = generator_preds_host if generator_all_preds is None else generator_all_preds + generator_preds_host
                if reranker_preds_host is not None:
                    reranker_all_preds = reranker_preds_host if reranker_all_preds is None else reranker_all_preds + reranker_preds_host
                if labels_host is not None:
                    all_labels = (
                        labels_host if all_labels is None else all_labels + labels_host
                    )

                # Set back to None to begin a new accumulation
                generator_losses_host, reranker_losses_host, generator_preds_host, reranker_preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if generator_losses_host is not None:
            generator_losses = nested_numpify(generator_losses_host)
            generator_all_losses = generator_losses if generator_all_losses is None else np.concatenate((generator_all_losses, generator_losses), axis=0)
        if reranker_losses_host is not None:
            reranker_losses = nested_numpify(reranker_losses_host)
            reranker_all_losses = reranker_losses if reranker_all_losses is None else np.concatenate((reranker_all_losses, reranker_losses), axis=0)
        if generator_preds_host is not None:
            generator_all_preds = generator_preds_host if generator_all_preds is None else generator_all_preds + generator_preds_host
        if reranker_preds_host is not None:
            reranker_all_preds = reranker_preds_host if reranker_all_preds is None else reranker_all_preds + reranker_preds_host
        if labels_host is not None:
            all_labels = labels_host if all_labels is None else all_labels + labels_host

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if generator_all_losses is not None:
            generator_all_losses = generator_all_losses[:num_samples]
        if reranker_all_losses is not None:
            reranker_all_losses = reranker_all_losses[:num_samples]
        if generator_all_preds is not None:
            generator_all_preds = generator_all_preds[:num_samples]
        if reranker_all_preds is not None:
            reranker_all_preds = reranker_all_preds[:num_samples]
        if all_labels is not None:
            all_labels = all_labels[:num_samples]

        # Metrics!
        if self.compute_metrics is not None and generator_all_preds is not None and all_labels is not None:
            generator_metrics = self.compute_metrics(EvalPrediction(predictions=generator_all_preds, label_ids=all_labels))
        else:
            generator_metrics = {}

        if self.compute_metrics is not None and reranker_all_preds is not None and all_labels is not None:
            reranker_metrics = self.compute_metrics(EvalPrediction(predictions=reranker_all_preds, label_ids=all_labels))
        else:
            reranker_metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        generator_metrics = denumpify_detensorize(generator_metrics)
        reranker_metrics = denumpify_detensorize(reranker_metrics)

        if generator_all_losses is not None:
            generator_metrics[f"{metric_key_prefix}_loss"] = generator_all_losses.mean().item()

        if reranker_all_losses is not None:
            reranker_metrics[f"{metric_key_prefix}_loss"] = reranker_all_losses.mean().item()

        # in here we need to combine two metrics together...
        metrics = {}
        # Prefix all keys with metric_key_prefix + '_'
        if self.args.evaluate_generator:
            for key in list(generator_metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"generator_{metric_key_prefix}_{key}"] = generator_metrics[key]
                else:
                    metrics[f"generator_{key}"] = generator_metrics[key]
        for key in list(reranker_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"reranker_{metric_key_prefix}_{key}"] = reranker_metrics[key]
            else:
                metrics[f"reranker_{key}"] = reranker_metrics[key]

        return EvalLoopOutput_ours(generator_predictions=generator_all_preds, reranker_predictions=reranker_all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)



    def prediction_step(
        self,
        generator_model: nn.Module,
        reranker_model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        # has_labels = all(inputs.get(k) is not None for k in self.label_names)

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.reranker_model, "config"):
                reranker_ignore_keys = getattr(self.reranker_model.config, "keys_to_ignore_at_inference", [])
            else:
                reranker_ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        # if has_labels:
        #     labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
        #     if len(labels) == 1:
        #         labels = labels[0]
        # else:
        #     labels = None
        #predict for generator
        # we use beam search when evaluating generator
        if self.args.evaluate_generator or self.args.generate_eval_candidates:
            assert self.args.generate_candidate_strategy in ['beamsearch', 'group', 'sampling']
            if self.args.generate_candidate_strategy == 'beamsearch':
                gen_kwargs = {
                    "max_length": self.args.generator_max_target_length,
                    "min_length": self.args.generation_min_length,
                    "num_beams": self.args.num_cand_generated,
                    "num_return_sequences": self.args.num_cand_generated,
                    "no_repeat_ngram_size": self.args.generation_no_repeat_ngram_size,
                    "synced_gpus": True if is_deepspeed_zero3_enabled() else False
                }
            elif self.args.generate_candidate_strategy == 'group':
                # group beam search
                gen_kwargs = {
                    "max_length": self.args.generator_max_target_length,
                    "min_length": self.args.generation_min_length,
                    "num_beams": self.args.num_cand_generated,
                    "num_beam_groups": self.args.num_cand_generated,
                    "diversity_penalty": 1.0,
                    "no_repeat_ngram_size": self.args.generation_no_repeat_ngram_size,
                    "num_return_sequences": self.args.num_cand_generated,
                    "synced_gpus": True if is_deepspeed_zero3_enabled() else False
                }
            elif self.args.generate_candidate_strategy == 'sampling':
                # group beam search
                gen_kwargs = {
                    "max_length": self.args.generator_max_target_length,
                    "min_length": self.args.generation_min_length,
                    "num_beams": 1,
                    "do_sample": True,
                    "no_repeat_ngram_size": self.args.generation_no_repeat_ngram_size,
                    "num_return_sequences": self.args.num_cand_generated,
                    "synced_gpus": True if is_deepspeed_zero3_enabled() else False
                }

            generated_tokens = self.generator_model.generate(
                inputs["generator_input_ids"],
                attention_mask=inputs["generator_attention_mask"],
                **gen_kwargs,
            )

            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

            generated_seqs = self.generator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            # pick the first on of each seqs as the final generated results of generator
            generator_preds = []
            for i in range(inputs['generator_input_ids'].size(0)):
                generator_preds.append(generated_seqs[i*self.args.num_cand_generated])
        else:
            generator_preds = None

        source_text = inputs.pop('source_text')
        target_texts = inputs.pop('target_text')
        
        if self.args.generate_eval_candidates:
            # for compute loss of reranker, we need to prepare the reranker inputs here
            # 1. get the candidates 2. get the reranker input
            # _, candidate_texts,_ = self.compute_metrics.get_candidates(target_texts, generated_seqs, 
            #                     self.args.num_cand_generated, self.args.num_cand_generated, 'bottom') # (B* (C-1)), list with length of B*C
            reranker_input_ids, reranker_attention_mask = self._get_reranker_input_ids(source_text, generated_seqs)
            # get candidates for reranker to select
            candidates = [generated_seqs[i*self.args.num_cand_generated: (i+1)*self.args.num_cand_generated] for i in range(len(source_text))]
            reranker_inputs = {
                'reranker_input_ids': reranker_input_ids,
                'reranker_attention_mask': reranker_attention_mask,
            }
        else:
            reranker_inputs = inputs
            candidates = inputs.pop('candidates')

        
        with torch.no_grad():
            # for reranker
            if is_sagemaker_mp_enabled():
                reranker_raw_outputs = smp_forward_only(reranker_model, reranker_inputs)
                if isinstance(reranker_raw_outputs, dict):
                    reranker_loss_mb = reranker_raw_outputs["loss"]
                    reranker_logits_mb = tuple(v for k, v in reranker_raw_outputs.items() if k not in reranker_ignore_keys + ["loss"])
                else:
                    reranker_loss_mb = reranker_raw_outputs[0]
                    reranker_logits_mb = reranker_raw_outputs[1:]

                reranker_loss = reranker_loss_mb.reduce_mean().detach().cpu()
                reranker_logits = smp_nested_concat(reranker_logits_mb)

            else:
                reranker_loss, reranker_outputs = self.compute_loss_reranker_eval(reranker_model, reranker_inputs, return_outputs=True)
                reranker_loss = reranker_loss.mean().detach()
                if isinstance(reranker_outputs, dict):
                    reranker_logits = tuple(v for k, v in reranker_outputs.items() if k not in reranker_ignore_keys + ["loss"])
                else:
                    reranker_logits = reranker_outputs[1:]
            # for generator
            if self.args.evaluate_generator:
                if self.use_amp:
                    with autocast():
                        generator_outputs = generator_model(input_ids = inputs['generator_input_ids'], 
                                            attention_mask = inputs['generator_attention_mask'],
                                            labels = inputs['generator_labels'])
                else:
                    generator_outputs = generator_model(input_ids = inputs['generator_input_ids'], 
                                    attention_mask = inputs['generator_attention_mask'],
                                    labels = inputs['generator_labels'])
                if inputs['generator_labels'] is not None:
                    if self.label_smoother is not None:
                        generator_loss = self.label_smoother(generator_outputs, inputs["labels"]).mean().detach()
                    else:
                        generator_loss = (generator_outputs["loss"] if isinstance(generator_outputs, dict) else generator_outputs[0]).mean().detach()
                else:
                    generator_loss = None
            else:
                generator_loss = None

        if prediction_loss_only:
            return (generator_loss, reranker_loss, None, None, None)

        # predict for reranker
        reranker_logits = nested_detach(reranker_logits) # logits, hidden state, attention
        # if len(logits) == 1:
        # logits = logits[0].cpu().numpy() # (B, C)
        reranker_logits = reranker_logits[0]

        # select_ids = np.argmax(logits, axis=1) # (B,)
        select_ids = torch.argmax(reranker_logits, dim = 1) # (B,)

        targets = []
        reranker_preds = []
        for i,t in enumerate(target_texts):
            targets.append(t)
            reranker_preds.append(candidates[i][select_ids[i]])

        return (generator_loss, reranker_loss, generator_preds, reranker_preds, targets)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.generator_tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenizer was passed when creating "
                "this `Trainer`. Make sure to create your `Trainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            self.generator_tokenizer.pad_token_id if self.generator_tokenizer.pad_token_id is not None else self.generator_tokenizer.eos_token_id
        )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

