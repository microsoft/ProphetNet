#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
from operator import mod
import os
import random
import json
import sys
from dataclasses import dataclass, field
from typing import Optional
from xml.parsers.expat import model

import numpy as np
from datasets import load_dataset, load_metric
from data_utils.dataset import ReRankingDataset
from model_utils.reranker_utils import RobertaRanker, LongformerRanker
from model_utils.generation_utils import BartForConditionalGeneration
from model_utils.utils import load_reranker_model, load_generator_model
from trainer_utils.trainer import Trainer
from data_utils.data_collator import DataCollator_train, DataCollator_eval

from data_utils.metric_utils import compute_rouge, compute_coqa, compute_dialog, compute_qg, clean
from trainer_utils.training_args import TrainingArguments

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    default_data_collator,
    set_seed,
)
from transformers import RobertaConfig, RobertaTokenizer, LongformerConfig, LongformerTokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


try:
    nltk.data.find("omw-1.4")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("omw-1.4", quiet=True)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default="sum",
        metadata={"help": "sum/qg/dialog/coqa"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )

    use_untokenized_data : Optional[bool] = field(
        default=False,
        metadata={
            "help": "use the untokenized data (used when the preprocessed data is tokenized by unsuitable tokenizer)"
        },
    )

    cache_data : Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to cache data in memory, useful when training on cloud with blob container"
        },
    )

    generator_max_source_length : Optional[int] = field(
        default=1020,
        metadata={
            "help": "max source length for generator"
        },
    )

    reranker_max_source_length : Optional[int] = field(
        default=400,
        metadata={
            "help": "max source length for reranker"
        },
    )

    reranker_max_target_length : Optional[int] = field(
        default=100,
        metadata={
            "help": "max candidate length"
        },
    )

    generator_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "max candidate length"
        },
    )

    train_data_path: Optional[str] = field(
        default=None, metadata={"help": "The data path for training data, should not consist the file name"}
    )

    dev_data_path: Optional[str] = field(
        default=None, metadata={"help": "The data path for training data, should not consist the file name"}
    )

    test_data_path: Optional[str] = field(
        default=None, metadata={"help": "The data path for training data, should not consist the file name"}
    )

    load_tokenized_data: Optional[bool] = field(
        default=True, metadata={"help": "whether to load the preprocessed data, this will speed up the data loading stage"}
    )


    generate_eval_candidates:  Optional[bool] = field(
        default=True, metadata={"help": "whether to generate candidates with the co-trained generator when evaluation"}
    )


    # train_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the training data."}
    # )
    # validation_file: Optional[str] = field(
    #     default=None, metadata={"help": "A csv or a json file containing the validation data."}
    # )
    # test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    # def __post_init__(self):
    #     if self.task_name is not None:
    #         self.task_name = self.task_name.lower()
    #         if self.task_name not in task_to_keys.keys():
    #             raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
    #     elif self.dataset_name is not None:
    #         pass
    #     elif self.train_file is None or self.validation_file is None:
    #         raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
    #     else:
    #         train_extension = self.train_file.split(".")[-1]
    #         assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #         validation_extension = self.validation_file.split(".")[-1]
    #         assert (
    #             validation_extension == train_extension
    #         ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    reranker_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    generator_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    loss_type: str = field(
        default="contrastive",
        metadata={"help": "use ranking loss or binary loss"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    position_extend_way: str = field(
        default='normal',
        metadata={
            "help": "to initialize the new position embedding weights from normal (normal) "
            "or copying from trained position embedding of the original model (copys)"
        },
    )
    reranker_model_type: str = field(
        default='roberta',
        metadata={
            "help": "reranker base model type: roberta or longformer"
        },
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    reranker_config, reranker_tokenizer, reranker_model = load_reranker_model(model_args, data_args)
    generator_config, generator_tokenizer, generator_model = load_generator_model(model_args, data_args)
    
    
    if training_args.do_train:
        if data_args.dataset_name is None and data_args.train_data_path is None:
            raise ValueError("There should be either dataset_name or train_data_path")
        if data_args.train_data_path is not None:
            data_dir = data_args.train_data_path
        else:
            data_dir = data_args.dataset_name
        train_dataset = ReRankingDataset(data_dir,  generator_tokenizer=generator_tokenizer, reranker_tokenizer=reranker_tokenizer,split='train', args = data_args, shuffle = True, is_train=True)
        

    if training_args.do_eval:
        if data_args.dataset_name is None and data_args.dev_data_path is None:
            raise ValueError("There should be either dataset_name or dev_data_path")
        if data_args.dev_data_path is not None:
            data_dir = data_args.dev_data_path
        else:
            data_dir = data_args.dataset_name      
        eval_dataset = ReRankingDataset(data_dir, generator_tokenizer=generator_tokenizer, reranker_tokenizer=reranker_tokenizer, split='dev', args = data_args)
        

    if training_args.do_predict or training_args.do_eval:
        if data_args.dataset_name is None and data_args.test_data_path is None:
            raise ValueError("There should be either dataset_name or dev_data_path")
        if data_args.test_data_path is not None:
            data_dir = data_args.test_data_path
        else:
            data_dir = data_args.dataset_name
        test_dataset = ReRankingDataset(data_dir, generator_tokenizer=generator_tokenizer, reranker_tokenizer=reranker_tokenizer, split='test', args = data_args)
        


    # Log a few random samples from the training set:
    # if training_args.do_train:
    #     for index in random.sample(range(len(train_dataset)), 3):
    #         logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function

    if data_args.task_name == "sum":
        compute_metrics = compute_rouge()
    elif data_args.task_name == 'qg':
        compute_metrics = compute_qg()
    elif data_args.task_name == 'coqa':
        compute_metrics = compute_coqa()
    elif data_args.task_name == 'dialog':
        compute_metrics = compute_dialog()

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.


    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    # if data_args.pad_to_max_length:
    #     data_collator = default_data_collator
    # elif training_args.fp16:
    #     data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    # else:
    #     data_collator = None

    data_collator = DataCollator_train(generator_tokenizer)
    data_collator_eval = DataCollator_eval(generator_tokenizer, reranker_tokenizer, data_args.generate_eval_candidates)

    # passing needed trainer args for generation
    setattr(training_args, "reranker_loss_type", model_args.loss_type)
    setattr(training_args, "reranker_max_target_length", data_args.reranker_max_target_length)
    setattr(training_args, "generator_max_target_length", data_args.generator_max_target_length)
    setattr(training_args, "generate_eval_candidates", data_args.generate_eval_candidates)
    setattr(training_args, 'reranker_max_source_length', data_args.reranker_max_source_length)

    # Initialize our Trainer
    trainer = Trainer(
        generator_model=generator_model,
        reranker_model=reranker_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        generator_tokenizer=generator_tokenizer,
        reranker_tokenizer=reranker_tokenizer,
        data_collator=data_collator,
        data_collator_eval=data_collator_eval,
    )


    # Training
    if training_args.do_train:
        if training_args.training_mode == 'iterative':
            train_result = trainer.train()
        elif training_args.training_mode == 'co-train':
            pass
            train_result = trainer.co_train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        # trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_local_process_zero():
            trainer.metrics_tracker['evaluation_results'] = metrics

    if training_args.do_predict:
        logger.info("*** Predict ***")
        
        predict_results = trainer.predict(test_dataset, metric_key_prefix="predict")

        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        if trainer.is_local_process_zero():
            trainer.metrics_tracker['test_results'] = metrics

        if trainer.is_world_process_zero():
            generator_predictions = predict_results.generator_predictions
            generator_predictions = [clean(pred.strip()) for pred in generator_predictions]
            output_prediction_file = os.path.join(training_args.output_dir, "generator_generated_predictions.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(generator_predictions))

            reranker_predictions = predict_results.reranker_predictions
            reranker_predictions = [clean(pred.strip()) for pred in reranker_predictions]
            output_prediction_file = os.path.join(training_args.output_dir, "reranker_generated_predictions.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(reranker_predictions))
        # for predict_dataset, task in zip(predict_datasets, tasks):
        #     # Removing the `label` columns because it contains -1 and Trainer won't like that.
        #     predict_dataset.remove_columns_("label")
        #     predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        #     predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        #     output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
        #     if trainer.is_world_process_zero():
        #         with open(output_predict_file, "w") as writer:
        #             logger.info(f"***** Predict results {task} *****")
        #             writer.write("index\tprediction\n")
        #             for index, item in enumerate(predictions):
        #                 if is_regression:
        #                     writer.write(f"{index}\t{item:3.3f}\n")
        #                 else:
        #                     item = label_list[item]
        #                     writer.write(f"{index}\t{item}\n")

    # if trainer.is_local_process_zero():
    #     with open(os.path.join(training_args.output_dir, 'metrics_tracker.json'), 'w', encoding='utf-8') as f:
    #         json.dump(trainer.metrics_tracker, f)

    if training_args.do_train and trainer.is_local_process_zero():
        with open(os.path.join(training_args.output_dir, 'reward_tracker.json'), 'w', encoding='utf-8') as f:
            json.dump(trainer.reward_tracker, f)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
        if data_args.task_name is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "glue"
            kwargs["dataset_args"] = data_args.task_name
            kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()