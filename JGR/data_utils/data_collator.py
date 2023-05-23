import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass

from transformers.modeling_utils import PreTrainedModel

@dataclass
class DataCollator_train:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    # model: Optional[PreTrainedModel] = None
    # padding: Union[bool, str, PaddingStrategy] = True
    # max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # # same length to return tensors.
        # if labels is not None:
        #     max_label_length = max(len(l) for l in labels)
        #     padding_side = self.tokenizer.padding_side
        #     for feature in features:
        #         remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
        #         feature["labels"] = (
        #             feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
        #         )

        # features = self.tokenizer.pad(
        #     features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )

        # # prepare decoder_input_ids
        # if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
        #     decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
        #     features["decoder_input_ids"] = decoder_input_ids

        # input_ids for generator
        input_ids = pad_sequence([d['input_ids'] for d in features], batch_first = True, padding_value = self.tokenizer.pad_token_id) # (B, L )
        # labels for generator
        if features[0]['labels'] is not None:
            labels = pad_sequence([d['labels'] for d in features], batch_first = True, padding_value=self.label_pad_token_id)
        else:
            labels = None

    
        attention_mask = input_ids != self.tokenizer.pad_token_id

        batch = {}
        batch['input_ids'] = input_ids
        batch['labels'] = labels
        batch['attention_mask'] = attention_mask
        if features[0]['target_text'] is not None:
            batch['target_text'] = [d['target_text'] for d in features]
        batch['source_text'] = [d['source_text'] for d in features]
        # print(sample)
        # exit()
        return batch



@dataclass
class DataCollator_eval:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    generator_tokenizer: PreTrainedTokenizerBase
    reranker_tokenizer: PreTrainedTokenizerBase
    generate_eval_candidates: bool = False
    # model: Optional[PreTrainedModel] = None
    # padding: Union[bool, str, PaddingStrategy] = True
    # max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # # same length to return tensors.
        # if labels is not None:
        #     max_label_length = max(len(l) for l in labels)
        #     padding_side = self.tokenizer.padding_side
        #     for feature in features:
        #         remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
        #         feature["labels"] = (
        #             feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
        #         )

        # features = self.tokenizer.pad(
        #     features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )

        # # prepare decoder_input_ids
        # if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
        #     decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
        #     features["decoder_input_ids"] = decoder_input_ids

        # for reranker
        candidate_ids = []
        batch_size = len(features)

        # for generator
        input_ids = pad_sequence([d['input_ids'] for d in features], batch_first = True, padding_value = self.generator_tokenizer.pad_token_id) # (B, L )
        generator_attention_mask = input_ids != self.generator_tokenizer.pad_token_id

        batch = {}

        batch['generator_input_ids'] = input_ids
        batch['generator_attention_mask'] = generator_attention_mask

        if features[0]['labels'] is not None:
            batch['generator_labels'] = pad_sequence([d['labels'] for d in features], batch_first = True, padding_value=self.label_pad_token_id)
        else:
            batch['generator_labels'] = None
        
        if features[0]['target_text'] is not None:
            batch['target_text'] = [d['target_text'] for d in features]
        else:
            batch['target_text'] = None
        
        
        batch['source_text'] = [d['source_text'] for d in features]

        # print(sample)
        # exit()
        return batch



