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
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from logging import StreamHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

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
)
from transformers.training_args import ParallelMode, TrainingArguments
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

def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
    if not self.args.remove_unused_columns:
        return dataset
    if self._signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(self.model.forward)
        self._signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        self._signature_columns += ["label", "label_ids"]
    columns = [k for k in self._signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
    if len(ignored_columns) > 0:
        dset_description = "" if description is None else f"in the {description} set "
        logger.info(
            f"The following columns {dset_description} don't have a corresponding argument in "
            f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
        )

    if version.parse(datasets.__version__) < version.parse("1.4.0"):
        dataset.set_format(
            type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
        )
        return dataset
    else:
        return dataset.remove_columns(ignored_columns)

def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
    if not isinstance(self.train_dataset, collections.abc.Sized):
        return None

    generator = None
    if self.args.world_size <= 1 and _is_torch_generator_available:
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

    # Build the sampler.
    if self.args.group_by_length:
        if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
            lengths = (
                self.train_dataset[self.args.length_column_name]
                if self.args.length_column_name in self.train_dataset.column_names
                else None
            )
        else:
            lengths = None
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        if self.args.world_size <= 1:
            return LengthGroupedSampler(
                self.train_dataset,
                self.args.train_batch_size,
                lengths=lengths,
                model_input_name=model_input_name,
                generator=generator,
            )
        else:
            return DistributedLengthGroupedSampler(
                self.train_dataset,
                self.args.train_batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                lengths=lengths,
                model_input_name=model_input_name,
                seed=self.args.seed,
            )

    else:
        if self.args.world_size <= 1:
            if _is_torch_generator_available:
                return RandomSampler(self.train_dataset, generator=generator)
            return RandomSampler(self.train_dataset)
        elif (
            self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
            and not self.args.dataloader_drop_last
        ):
            # Use a loop for TPUs when drop_last is False to have all batches have the same size.
            return DistributedSamplerWithLoop(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
            )
        else:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
            )

def get_train_dataloader(self) -> DataLoader:
    """
    Returns the training :class:`~torch.utils.data.DataLoader`.
    Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
    to distributed training if necessary) otherwise.
    Subclass and override this method if you want to inject some custom behavior.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = self._remove_unused_columns(train_dataset, description="training")

    if isinstance(train_dataset, torch.utils.data.dataset.IterableDataset):
        if self.args.world_size > 1:
            train_dataset = IterableDatasetShard(
                train_dataset,
                batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    train_sampler = self._get_train_sampler()

    return DataLoader(
        train_dataset,
        batch_size=self.args.train_batch_size,
        sampler=train_sampler,
        collate_fn=self.data_collator,
        drop_last=self.args.dataloader_drop_last,
        num_workers=self.args.dataloader_num_workers,
        pin_memory=self.args.dataloader_pin_memory,
    )

def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
    # Deprecated code
    if self.args.use_legacy_prediction_loop:
        if is_torch_tpu_available():
            return SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif is_sagemaker_mp_enabled():
            return SequentialDistributedSampler(
                eval_dataset,
                num_replicas=smp.dp_size(),
                rank=smp.dp_rank(),
                batch_size=self.args.per_device_eval_batch_size,
            )
        elif self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)

    if self.args.world_size <= 1:
        return SequentialSampler(eval_dataset)
    else:
        return ShardSampler(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_processes=self.args.world_size,
            process_index=self.args.process_index,
        )

def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
    """
    Returns the evaluation :class:`~torch.utils.data.DataLoader`.
    Subclass and override this method if you want to inject some custom behavior.
    Args:
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
            accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
    """
    if eval_dataset is None and self.eval_dataset is None:
        raise ValueError("Trainer: evaluation requires an eval_dataset.")
    eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

    if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
        eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

    if isinstance(eval_dataset, torch.utils.data.dataset.IterableDataset):
        if self.args.world_size > 1:
            eval_dataset = IterableDatasetShard(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                drop_last=self.args.dataloader_drop_last,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator_eval,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    eval_sampler = self._get_eval_sampler(eval_dataset)

    return DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=self.args.eval_batch_size,
        collate_fn=self.data_collator_eval,
        drop_last=self.args.dataloader_drop_last,
        num_workers=self.args.dataloader_num_workers,
        pin_memory=self.args.dataloader_pin_memory,
    )

def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
    """
    Returns the test :class:`~torch.utils.data.DataLoader`.
    Subclass and override this method if you want to inject some custom behavior.
    Args:
        test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
    """
    if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
        test_dataset = self._remove_unused_columns(test_dataset, description="test")

    if isinstance(test_dataset, torch.utils.data.dataset.IterableDataset):
        if self.args.world_size > 1:
            test_dataset = IterableDatasetShard(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                drop_last=self.args.dataloader_drop_last,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )
        return DataLoader(
            test_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator_eval,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    test_sampler = self._get_eval_sampler(test_dataset)

    # We use the same batch_size as for eval.
    return DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=self.args.eval_batch_size,
        collate_fn=self.data_collator_eval,
        drop_last=self.args.dataloader_drop_last,
        pin_memory=self.args.dataloader_pin_memory,
    )

def create_optimizer(self, optimizer, model, learning_rate):
    """
    Setup the optimizer.
    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
    if optimizer is None:
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = learning_rate
        
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    if is_sagemaker_mp_enabled():
        optimizer = smp.DistributedOptimizer(optimizer)
    return optimizer

def create_scheduler(self, lr_scheduler, optimizer, num_training_steps: int):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.
    Args:
        num_training_steps (int): The number of training steps to do.
    """
    if lr_scheduler is None:
        warmup_steps = (
            self.args.warmup_steps
            if self.args.warmup_steps > 0
            else math.ceil(num_training_steps * self.args.warmup_ratio)
        )

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    return lr_scheduler

def num_examples(self, dataloader: DataLoader) -> int:
    """
    Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.
    Will raise an exception if the underlying dataset does not implement method :obj:`__len__`
    """
    return len(dataloader.dataset)


def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
    """
    For models that inherit from :class:`~transformers.PreTrainedModel`, uses that method to compute the number of
    floating point operations for every backward + forward pass. If using another model, either implement such a
    method in the model or subclass and override this method.
    Args:
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
    Returns:
        :obj:`int`: The number of floating-point operations.
    """
    flos = 0
    if hasattr(self.generator_model, "floating_point_ops"):
        flos += self.generator_model.floating_point_ops(inputs)
    if hasattr(self.reranker_model, "floating_point_ops"):
        flos += self.reranker_model.floating_point_ops(inputs)
    return flos


def create_model_card(
    self,
    language: Optional[str] = None,
    license: Optional[str] = None,
    tags: Optional[str] = None,
    model_name: Optional[str] = None,
    finetuned_from: Optional[str] = None,
    tasks: Optional[str] = None,
    dataset_tags: Optional[Union[str, List[str]]] = None,
    dataset: Optional[Union[str, List[str]]] = None,
    dataset_args: Optional[Union[str, List[str]]] = None,
):
    training_summary = TrainingSummary.from_trainer(
        self,
        language=language,
        license=license,
        tags=tags,
        model_name=model_name,
        finetuned_from=finetuned_from,
        tasks=tasks,
        dataset_tags=dataset_tags,
        dataset=dataset,
        dataset_args=dataset_args,
    )
    model_card = training_summary.to_model_card()
    with open(os.path.join(self.args.output_dir, "README.md"), "w") as f:
        f.write(model_card)

def _gather_and_numpify(self, tensors, name):
    """
    Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
    concatenating them to `gathered`
    """
    if tensors is None:
        return
    if is_torch_tpu_available():
        tensors = nested_xla_mesh_reduce(tensors, name)
    elif is_sagemaker_mp_enabled():
        tensors = smp_gather(tensors)
    elif self.args.local_rank != -1:
        tensors = distributed_concat(tensors)

    return nested_numpify(tensors)

def _nested_gather(self, tensors, name=None):
    """
    Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
    concatenating them to `gathered`
    """
    if tensors is None:
        return
    if is_torch_tpu_available():
        if name is None:
            name = "nested_gather"
        tensors = nested_xla_mesh_reduce(tensors, name)
    elif is_sagemaker_mp_enabled():
        tensors = smp_gather(tensors)
    elif self.args.local_rank != -1:
        tensors = distributed_concat(tensors)
    return tensors

def _nested_gather_object(self, objects, name=None):
    """
    Gather value of python objects
    """
    if objects is None:
        return
    if self.args.local_rank != -1:
        outputs = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(outputs, objects)
        objects = []
        for o in outputs:
            objects += o

    return objects

# Copied from Accelerate.
def _pad_across_processes(self, tensor, pad_index=-100):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
    they can safely be gathered.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(self._pad_across_processes(t, pad_index=pad_index) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: self._pad_across_processes(v, pad_index=pad_index) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )

    if len(tensor.shape) < 2:
        return tensor
    # Gather all sizes
    size = torch.tensor(tensor.shape, device=tensor.device)[None]
    sizes = self._nested_gather(size).cpu()

    max_size = max(s[1] for s in sizes)
    if tensor.shape[1] == max_size:
        return tensor

    # Then pad to the maximum size
    old_size = tensor.shape
    new_size = list(old_size)
    new_size[1] = max_size
    new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
    new_tensor[:, : old_size[1]] = tensor
    return new_tensor


def store_flos(self):
    # Storing the number of floating-point operations that went into the model
    if self.args.local_rank != -1:
        self.total_flos += distributed_broadcast_scalars([self.current_flos]).sum().item()
        self.current_flos = 0
    else:
        self.total_flos += self.current_flos
        self.current_flos = 0

def _sorted_checkpoints(
    self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    # Make sure we don't delete the best model.
    if self.best_model_checkpoint is not None:
        best_model_index = checkpoints_sorted.index(str(Path(self.best_model_checkpoint)))
        for i in range(best_model_index, len(checkpoints_sorted) - 2):
            checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
    return checkpoints_sorted

def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
    if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
    if len(checkpoints_sorted) <= self.args.save_total_limit:
        return

    # If save_total_limit=1 with load_best_mode_at_end=True, we could end up deleting the last checkpoint, which
    # we don't do to allow resuming.
    save_total_limit = self.args.save_total_limit
    if (
        self.best_model_checkpoint is not None
        and self.args.save_total_limit == 1
        and checkpoints_sorted[-1] != self.best_model_checkpoint
    ):
        save_total_limit = 2

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint)


def is_local_process_zero(self) -> bool:
    """
    Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
    machines) main process.
    """
    return self.args.local_process_index == 0

def is_world_process_zero(self) -> bool:
    """
    Whether or not this process is the global main process (when training in a distributed fashion on several
    machines, this is only going to be :obj:`True` for one process).
    """
    # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
    # process index.
    if is_sagemaker_mp_enabled():
        return smp.rank() == 0
    else:
        return self.args.process_index == 0

def save_model(self, output_dir: Optional[str] = None):
    """
    Will save the model, so you can reload it using :obj:`from_pretrained()`.
    Will only save from the main process.
    """

    if output_dir is None:
        output_dir = self.args.output_dir

    if self.is_world_process_zero():
        g_output_dir = os.path.join(output_dir,"generator")
        r_output_dir = os.path.join(output_dir,"reranker")
        # If we are executing this function, we are the process zero, so we don't check for that.
        # output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(g_output_dir, exist_ok=True)
        os.makedirs(r_output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        # save generator
        if not isinstance(self.generator_model, PreTrainedModel):
            if isinstance(unwrap_model(self.generator_model), PreTrainedModel):
                state_dict = self.generator_model.state_dict()
                unwrap_model(self.generator_model).save_pretrained(g_output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                state_dict = self.generator_model.state_dict()
                torch.save(state_dict, os.path.join(g_output_dir, WEIGHTS_NAME))
        else:
            self.generator_model.save_pretrained(g_output_dir)
        if self.generator_tokenizer is not None:
            self.generator_tokenizer.save_pretrained(g_output_dir)
        
        # save reranker
        if not isinstance(self.reranker_model, PreTrainedModel):
            if isinstance(unwrap_model(self.reranker_model), PreTrainedModel):
                state_dict = self.reranker_model.state_dict()
                unwrap_model(self.reranker_model).save_pretrained(r_output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                state_dict = self.reranker_model.state_dict()
                torch.save(state_dict, os.path.join(r_output_dir, WEIGHTS_NAME))
        else:
            self.reranker_model.save_pretrained(r_output_dir)
        if self.reranker_tokenizer is not None:
            self.reranker_tokenizer.save_pretrained(r_output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def _save_checkpoint(self, metrics=None):
    # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    # want to save except FullyShardedDDP.
    # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    # Save model checkpoint
    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"

    run_dir = self.args.output_dir
    self.store_flos()

    output_dir = os.path.join(run_dir, checkpoint_folder)
    self.save_model(output_dir)

    # Determine the new best metric / best model checkpoint
    if metrics is not None and self.args.metric_for_best_model is not None:
        metric_to_check = self.args.metric_for_best_model
        # if not metric_to_check.startswith("eval_"):
        #     metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics[metric_to_check]

        operator = np.greater if self.args.greater_is_better else np.less
        if (
            self.best_metric is None
            or self.best_model_checkpoint is None
            or operator(metric_value, self.best_metric)
        ):
            self.best_metric = metric_value
            self.best_model_checkpoint = output_dir


    # # Maybe delete some older checkpoints.
    if self.is_world_process_zero():
        self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

def _load_optimizer_and_scheduler(self, checkpoint):
    """If optimizer and scheduler states exist, load them."""
    if checkpoint is None:
        return


    if os.path.isfile(os.path.join(checkpoint, "optimizer.pt")) and os.path.isfile(
        os.path.join(checkpoint, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        if is_torch_tpu_available():
            # On TPU we have to take some extra precautions to properly load the states on the right device.
            optimizer_state = torch.load(os.path.join(checkpoint, "optimizer.pt"), map_location="cpu")
            with warnings.catch_warnings(record=True) as caught_warnings:
                lr_scheduler_state = torch.load(os.path.join(checkpoint, "scheduler.pt"), map_location="cpu")
            reissue_pt_warnings(caught_warnings)

            xm.send_cpu_data_to_device(optimizer_state, self.args.device)
            xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

            self.optimizer.load_state_dict(optimizer_state)
            self.lr_scheduler.load_state_dict(lr_scheduler_state)
        else:
            map_location = "cpu" if is_sagemaker_mp_enabled() else self.args.device
            self.optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint, "optimizer.pt"), map_location=map_location)
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, "scheduler.pt")))
            reissue_pt_warnings(caught_warnings)
            if self.use_amp and os.path.isfile(os.path.join(checkpoint, "scaler.pt")):
                self.scaler.load_state_dict(torch.load(os.path.join(checkpoint, "scaler.pt")))


def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            kwargs = dict(device=self.args.device)
            inputs[k] = v.to(**kwargs)

    if self.args.past_index >= 0 and self._past is not None:
        inputs["mems"] = self._past

    return inputs


def _tune_save_checkpoint(self):
    from ray import tune

    if not self.use_tune_checkpoints:
        return
    with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
        output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        self.save_model(output_dir)
        if self.is_world_process_zero():
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

def call_model_init(self, trial=None):
    model_init_argcount = len(inspect.signature(self.model_init).parameters)
    if model_init_argcount == 0:
        model = self.model_init()
    elif model_init_argcount == 1:
        model = self.model_init(trial)
    else:
        raise RuntimeError("model_init should have 0 or 1 argument.")

    if model is None:
        raise RuntimeError("model_init should not return None.")

    return model

def _wrap_model(self, model, training=True):
    if is_sagemaker_mp_enabled():
        # Wrapping the base model twice in a DistributedModel will raise an error.
        if isinstance(self.model_wrapped, smp.model.DistributedModel):
            return self.model_wrapped
        return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)


    # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
    if unwrap_model(model) is not model:
        return model

    # Mixed precision training with apex (torch < 1.6)
    if self.use_apex and training:
        model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization)
    if self.args.n_gpu > 1:
        model = nn.DataParallel(model)

    # Note: in torch.distributed mode, there's no point in wrapping the model
    # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
    if not training:
        return model

    # Distributed training (should be after apex fp16 initializatio
    if is_sagemaker_dp_enabled():
        model = DDP(model, device_ids=[dist.get_local_rank()], broadcast_buffers=False)
    elif self.args.local_rank != -1:
        if self.args.ddp_find_unused_parameters is not None:
            find_unused_parameters = self.args.ddp_find_unused_parameters
        elif isinstance(model, PreTrainedModel):
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
            find_unused_parameters = not getattr(model.config, "gradient_checkpointing", False)
        else:
            find_unused_parameters = True
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=find_unused_parameters,
        )

    return model

def _load_state_dict_in_model(self, generator_state_dict, reranker_state_dict):
    generator_load_result = self.generator_model.load_state_dict(generator_state_dict, strict=False)

    if len(generator_load_result.missing_keys) != 0:
        if set(generator_load_result.missing_keys) == set(self.generator_model._keys_to_ignore_on_save):
            self.generator_model.tie_weights()
        else:
            logger.warn(f"There were missing keys in the checkpoint model loaded: {generator_load_result.missing_keys}.")
    if len(generator_load_result.unexpected_keys) != 0:
        logger.warn(f"There were unexpected keys in the checkpoint model loaded: {generator_load_result.unexpected_keys}.")

    reranker_load_result = self.reranker_model.load_state_dict(reranker_state_dict, strict=False)

    if len(reranker_load_result.missing_keys) != 0:
        if set(reranker_load_result.missing_keys) == set(self.reranker_model._keys_to_ignore_on_save):
            self.reranker_model.tie_weights()
        else:
            logger.warn(f"There were missing keys in the checkpoint model loaded: {reranker_load_result.missing_keys}.")
    if len(reranker_load_result.unexpected_keys) != 0:
        logger.warn(f"There were unexpected keys in the checkpoint model loaded: {reranker_load_result.unexpected_keys}.")

def _get_learning_rate(self, lr_scheduler):
    last_lr = (
        # backward compatibility for pytorch schedulers
        lr_scheduler.get_last_lr()[0]
        if version.parse(torch.__version__) >= version.parse("1.4")
        else lr_scheduler.get_lr()[0]
    )
    return last_lr