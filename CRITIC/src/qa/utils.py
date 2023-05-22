import string
import random
import re
import time
import json
from collections import Counter
from typing import TypeVar, Iterable, List, Union, Any
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score


def extract_cot_answer(cot):
    if not cot:
        return ""

    # get answer
    cot = cot.strip(" ")
    cot = cot.split("<|endoftext|>")[0]  # text-davinci-003
    TEMPLATE = "is: "
    if TEMPLATE not in cot:
        return ""

    start_idx = cot.rfind(TEMPLATE) + len(TEMPLATE)
    end_idx = -1 if cot.endswith(".") else len(cot)
    ans_span = cot[start_idx: end_idx].strip()

    return ans_span


def is_null_answer(text):
    if not text:
        return True
    text = text.strip().lower()
    if  text in ["none", "", "no answer", "never", "null", "both", "neither"]:
        return True
    if text.startswith("none"):
        return True
    return False


def get_end_index(tokens, end_tokens=["\n", "<|endoftext|>"], verbose=True):
    stop_token = None
    for end_tk in end_tokens:
        for tk in tokens:
            if end_tk in tk:
                stop_token = tk
                break
    if not stop_token:
        end_idx = len(tokens)
    else:
        end_idx = tokens.index(stop_token)
    return end_idx


def normalize_answer(s):

    def replace_ordinals(s):
        ordinal_map = {
            "first": "1",
            "second": "2",
            "third": "3",
            "fourth": "4",
            "fifth": "5",
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            # more as needed
        }
        for ordinal, number in ordinal_map.items():
            s = s.replace(ordinal, number)
        return s

    def normalize_yes_no(s):
        mp = {
            "true": "yes",
            "false": "no",
        }
        return mp.get(s, s)
    
    def remove_rank(text):
        return re.sub(r"(?<=\w)(st|nd|rd|th)\b", "", text)

    def remove_articles(text):
        if len(text.split(" ")) > 1:
            text = re.sub(r"\b(a|an|the)\b", " ", text)
        # remove 's' in the end of word
        text = " ".join([t.rstrip("s") for t in text.split(" ")])
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`", u"–"]))
        for c in exclude:
            text = text.replace(c, " ")
        return text

    def lower(text):
        return text.lower()
    
    def normalize_number(a):
        if a[-1] == ".":
                a = a[:-1]
        if a[-2:] == ".0":
            a = a[:-2]
        return a

    return normalize_yes_no(remove_rank(replace_ordinals(white_space_fix(remove_articles(remove_punc(lower(s)))))))


def em_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    em = (normalized_prediction == normalized_ground_truth)

    ZERO_METRIC = (0, 0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return em, f1, precision, recall


def multi_ref_score(prediction, ground_truth):
    """
    ground_truth: list or str
        - multiple references split by "; "
    """
    if isinstance(ground_truth, str):
        ground_truth = ground_truth.split("; ")
    scores = [em_f1_score(prediction, gt)[:2] for gt in ground_truth]
    scores.append(em_f1_score(prediction, " ".join(ground_truth))[:2]) # may predict multi answer
    # scores = sorted(scores, reverse=True)
    return max(scores)


if __name__ == "__main__":
    print(em_f1_score("Cabrini-Green Projects", "Cabrini–Green projects"))
    print(em_f1_score("January 6th", "January 6"))
    print(em_f1_score("October 25, 1922", "October 1922"))
