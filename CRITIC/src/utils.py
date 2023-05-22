import os
import json
import random
import time
import re
import string
import numpy as np
import requests
from pathlib import Path
from typing import TypeVar, Iterable, List, Union, Any


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_prompt(data, prompt_type):
    prompt_path = "./prompts/{}/{}.txt".format(data, prompt_type)
    if not os.path.exists(prompt_path):
        prompt_path = prompt_path.replace(".txt", ".md")
    with open(prompt_path, 'r', encoding='utf-8') as fp:
        prompt = fp.read().strip() + "\n\n"
    return prompt


def list_rindex(alist, value):
    # assert value in alist, alist
    if value not in alist:
        print(f">> rindex error: {value} not in {alist}")
        return len(alist)
    return len(alist) - alist[-1::-1].index(value) -1
