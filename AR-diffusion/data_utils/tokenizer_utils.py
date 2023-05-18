import os
import json
import logging
import pathlib
import torch
from transformers import AutoTokenizer

from tokenizers.processors import BertProcessing
from tokenizers import ByteLevelBPETokenizer, decoders

logging.basicConfig(level=logging.INFO)

def create_tokenizer(path, return_pretokenized: bool = False, tokenizer_ckpt: str = None):
    
    if return_pretokenized:
        print(f'*******use pretrained tokenizer*****{return_pretokenized}*******')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
        return tokenizer

    return read_byte_level(path)


def train_bytelevel(
    path, #list
    save_path,
    vocab_size=10000,
    min_frequency=1,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>", ],  # replace <s> to <length>
):

    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(
        files=path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    tokenizer.save_model(str(pathlib.Path(save_path)))


def read_byte_level(path: str):
    tokenizer = ByteLevelBPETokenizer(
        f"{path}/vocab.json",
        f"{path}/merges.txt",
    )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),  # replace <s> to <length>
    )

    tokenizer.enable_truncation(max_length=512)

    with open(f"{path}/vocab.json", "r") as fin:
        vocab = json.load(fin)

    # add length method to tokenizer object
    tokenizer.vocab_size = len(vocab)

    # add length property to tokenizer object
    tokenizer.__len__ = property(lambda self: self.vocab_size)

    tokenizer.decoder = decoders.ByteLevel()
    print(tokenizer.vocab_size)

    print(
        tokenizer.encode(
            "Bores can be divided into two classes; those who have their own particular subject, and those who do not need a subject."
        ).ids
    )

    print(
        tokenizer.decode(
            tokenizer.encode(
                "Bores can be divided into two classes; those who have their own particular subject, and those who do not need a subject."
            ).ids,
            skip_special_tokens=True,
        )
    )

    ids = tokenizer.encode(
        "Bores can be divided into two classes; those who have their own particular subject, and those who do not need a subject."
    ).ids
    tensor = torch.tensor(ids)
    print(tokenizer.decode(tensor.tolist(), skip_special_tokens=True))
    print(f"Vocab size: {tokenizer.vocab_size}")

    return tokenizer


if __name__ == "__main__":
    path = './data/iwslt14/'
    vocab_size = 10000
    
    data_path = [path + item for item in os.listdir(path) if 'train' in item]
    train_bytelevel(path=data_path, vocab_size=vocab_size+5, save_path=path)

