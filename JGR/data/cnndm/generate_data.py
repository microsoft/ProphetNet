import hashlib
import os

from tqdm import tqdm

# this code is a modified version of code in 
# https://github.com/huggingface/datasets/blob/1.12.1/datasets/cnn_dailymail/cnn_dailymail.py

_DESCRIPTION = """\
CNN/DailyMail non-anonymized summarization dataset.
There are two features:
  - article: text of news article, used as the document to be summarized
  - highlights: joined text of highlights with <s> and </s> around each
    highlight, which is the target summary
"""

# The second citation introduces the source data, while the first
# introduces the specific form (non-anonymized) we use here.
_CITATION = """\
@article{DBLP:journals/corr/SeeLM17,
  author    = {Abigail See and
               Peter J. Liu and
               Christopher D. Manning},
  title     = {Get To The Point: Summarization with Pointer-Generator Networks},
  journal   = {CoRR},
  volume    = {abs/1704.04368},
  year      = {2017},
  url       = {http://arxiv.org/abs/1704.04368},
  archivePrefix = {arXiv},
  eprint    = {1704.04368},
  timestamp = {Mon, 13 Aug 2018 16:46:08 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/SeeLM17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@inproceedings{hermann2015teaching,
  title={Teaching machines to read and comprehend},
  author={Hermann, Karl Moritz and Kocisky, Tomas and Grefenstette, Edward and Espeholt, Lasse and Kay, Will and Suleyman, Mustafa and Blunsom, Phil},
  booktitle={Advances in neural information processing systems},
  pages={1693--1701},
  year={2015}
}
"""


def _get_hash_from_path(p):
    """Extract hash from path."""
    basename = os.path.basename(p)
    return basename[0 : basename.find(".story")]


def _find_files(dl_paths, publisher, url_dict):
    """Find files corresponding to urls."""
    if publisher == "cnn":
        top_dir = os.path.join(dl_paths["cnn_stories"], "cnn", "stories")
    elif publisher == "dm":
        top_dir = os.path.join(dl_paths["dm_stories"], "dailymail", "stories")
    else:
        print("Unsupported publisher: %s", publisher)
    files = sorted(os.listdir(top_dir))

    ret_files = []
    for p in files:
        if _get_hash_from_path(p) in url_dict:
            ret_files.append(os.path.join(top_dir, p))
    return ret_files

def _read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def _get_url_hashes(path):
    """Get hashes of urls in file."""
    urls = _read_text_file(path)

    def url_hash(u):
        h = hashlib.sha1()
        try:
            u = u.encode("utf-8")
        except UnicodeDecodeError:
            print("Cannot hash url: %s", u)
        h.update(u)
        return h.hexdigest()

    return {url_hash(u): True for u in urls}

def _subset_filenames(dl_paths, split):
    """Get filenames for a particular split."""
    assert isinstance(dl_paths, dict), dl_paths
    # Get filenames for a split.
    if split == 'train':
        urls = _get_url_hashes(dl_paths["train_urls"])
    elif split == 'val':
        urls = _get_url_hashes(dl_paths["val_urls"])
    elif split == 'test':
        urls = _get_url_hashes(dl_paths["test_urls"])
    else:
        print("Unsupported split: %s"%(split))
    cnn = _find_files(dl_paths, "cnn", urls)
    dm = _find_files(dl_paths, "dm", urls)
    return cnn + dm

def _get_art_abs(story_file, tfds_version):

    """Get abstract (highlights) and article from a story file path."""
    # Based on https://github.com/abisee/cnn-dailymail/blob/master/
    #     make_datafiles.py

    lines = _read_text_file(story_file)

    # The github code lowercase the text and we removed it in 3.0.0.

    # Put periods on the ends of lines that are missing them
    # (this is a problem in the dataset because many image captions don't end in
    # periods; consequently they end up in the body of the article as run-on
    # sentences)
    def fix_missing_period(line):
        """Adds a period to a line that is missing a period."""
        if "@highlight" in line:
            return line
        if not line:
            return line
        if line[-1] in END_TOKENS:
            return line
        return line + " ."

    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in lines:
        if not line:
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = " ".join(article_lines)

    if tfds_version >= "2.0.0":
        abstract = "\n".join(highlights)
    else:
        abstract = " ".join(highlights)

    return article, abstract

def generate_examples( files):
    samples = []
    for p in tqdm(files):
        article, highlights = _get_art_abs(p, "1.0.0")
        if article == "":
            continue
        if not article or not highlights:
            continue
        fname = os.path.basename(p)
        samples.append({
            'source': article,
            'target': highlights,
            "id": _get_hash_from_path(fname),
        })
    return samples


dl_paths = {
    # pylint: disable=line-too-long
    "cnn_stories": "raw_data/cnn_stories",
    "dm_stories": "raw_data/dailymail_stories",
    "test_urls": "raw_data/url_lists/all_test.txt",
    "train_urls": "raw_data/url_lists/all_train.txt",
    "val_urls": "raw_data/url_lists/all_val.txt",
    # pylint: enable=line-too-long
}

train_files = _subset_filenames(dl_paths, 'train')
dev_files = _subset_filenames(dl_paths, 'val')
test_files = _subset_filenames(dl_paths, 'test')

END_TOKENS = [".", "!", "?", "...", "'", "`", '"', "\u2019", "\u201d", ")"]




train_samples = generate_examples(train_files)
print("train data length:",len(train_samples))

import json
with open('train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_samples, f)


dev_samples = generate_examples(dev_files)
print("dev data length:",len(train_samples))
with open('dev_data.json', 'w', encoding='utf-8') as f:
    json.dump(dev_samples, f)


test_samples = generate_examples(test_files)
print("test data length:",len(train_samples))
with open('test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_samples, f)