# Modified based on: https://github.com/wenhuchen/Program-of-Thoughts/blob/main/tool.py
import re
from typing import Union, Any
from math import isclose
from sympy.solvers import solve
from sympy import Symbol, Eq
import math
from sympy import simplify
import numpy as np
import cvxpy as cp
import statistics


def normalize_answer(answer: str):
    answer = str(answer)
    # number
    answer = answer.replace(",", "")
    digits = re.findall(r"-?\d+\.?\d*", answer)
    answer = digits[-1] if len(digits) > 0 else None
    return floatify_ans(answer)


def get_precision(gt_ans: float) -> int:
    precision = 5
    if '.' in str(gt_ans):
        precision = len(str(gt_ans).split('.')[-1])
    return precision


def finqa_equal(prediction: Union[bool, float, str],
                reference: Union[float, str],
                include_percentage: bool = True,
                is_close: float = False) -> bool:
    if prediction is None:
        return False
    elif type(prediction) == bool:
        # bool questions
        if prediction:
            return reference == 'yes'
        else:
            return reference == 'no'
    elif type(reference) == str or type(prediction) == str:
        # string questions
        return prediction == reference
    else:
        # number questions
        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]
        for item in gt_result:
            try:
                if is_close:
                    if isclose(item, prediction, rel_tol=0.001):
                        return True
                precision = min(get_precision(prediction), get_precision(item))
                if round(prediction, precision) == round(item, precision):
                    return True
            except Exception:
                continue
        return False


def round_with_error(x):
    return round(x * 1e5) / 1e5


def floatify_ans(ans):
    """gsm8k"""
    if ans is None:
        return None
    elif type(ans) == dict:
        ans = list(ans.values())[0]
    elif type(ans) == bool:
        ans = ans
    elif type(ans) in [list, tuple]:
        if not ans:
            return None
        else:
            try:
                ans = float(ans[0])
            except Exception:
                ans = str(ans[0])
    else:
        try:
            ans = float(ans)
            ans = round_with_error(ans)
        except Exception:
            ans = str(ans)
    return ans


def parse_api_result(result):
    if not result or 'choices' not in result:
        return None

    to_return = [g['text'] for g in result['choices']]
    return to_return


# remove comment and empty lines in code
def remove_comment(code):
    code = code.split("\n")
    code = [line for line in code if not line.startswith("#")]
    code = [line for line in code if line.strip() != ""]
    return "\n".join(code)
