import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass

@dataclass
class DataCollatorForReranking:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
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
    """

    tokenizer: PreTrainedTokenizerBase
    model_type: str = "roberta"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        '''
        feature list of {
            "input_ids": [C, L]
        }
        '''
        max_cand_len = max([max([len(c) for c in x['input_ids']]) for x in features])

        def bert_pad(X, max_len=-1):
            if max_len < 0:
                max_len = max(len(x) for x in X)
            result = []
            for x in X:
                if len(x) < max_len:
                    x.extend([self.tokenizer.pad_token_id] * (max_len - len(x)))
                result.append(x)
            return torch.LongTensor(result)

        candidate_ids = [bert_pad(x['input_ids'], max_cand_len) for x in features]
        candidate_ids = torch.stack(candidate_ids) # (B, C, L)

        attention_mask = candidate_ids != self.tokenizer.pad_token_id

        batch = {
            'input_ids': candidate_ids,
            'attention_mask': attention_mask
        }

        if "data" in features[0].keys():
            batch['data'] = [x['data'] for x in features]  # {'source': untokenized sentence, "target": untokenized sentence, "candidates": list of untokenized sentence}

        return batch


