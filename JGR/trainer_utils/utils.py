
import copy
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

from transformers.file_utils import (
    ExplicitEnum,
    is_psutil_available,
    is_sagemaker_dp_enabled,
    is_tf_available,
    is_torch_available,
    is_torch_cuda_available,
    is_torch_tpu_available,
)

class TrainOutput(NamedTuple):
    global_step: int
    generator_training_loss: float
    reranker_training_loss: float
    metrics: Dict[str, float]


class EvalLoopOutput_ours(NamedTuple):
    generator_predictions: Union[np.ndarray, Tuple[np.ndarray]]
    reranker_predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput_ours(NamedTuple):
    generator_predictions: Union[np.ndarray, Tuple[np.ndarray]]
    reranker_predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
