import os
import sys
from sys import getsizeof

import argparse
from tqdm import tqdm
import time

import spacy
import torchtext
from torchtext.data import Field, Dataset, Example, Iterator, Batch
import torch
import numpy as np

import pickle
import mmap
"""
[TODO]
1. Consider using vocabulary from Manning's website
"""

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--batch-size", type=int, default=1e3,
        help="Write these many examples at a time to the file (otherwise OOM error)")
    parser.add_argument("--splits", type=str, default='train')
    parser.add_argument("--min-freq", type=int, default=1)

    return parser.parse_args()

args = load_args()
args.splits = args.splits.split(',')

INIT_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

TEXT_en = Field(
    sequential=True,
    use_vocab=True,
    init_token=INIT_TOKEN,
    eos_token=EOS_TOKEN,
    tokenize="spacy",
    tokenizer_language="en_core_web_sm",
    lower=True
)

TEXT_de = Field(
    sequential=True,
    use_vocab=True,
    init_token=INIT_TOKEN,
    eos_token=EOS_TOKEN,
    tokenize="spacy",
    tokenizer_language="de_core_news_sm",
    lower=True
)

device = torch.device('cpu')

def process(split='train'):

    en_fname = os.path.join(args.data_dir, f'{split}',  f'{split}.en')
    de_fname = os.path.join(args.data_dir, f'{split}', f'{split}.de')

    en_fields = ('src', TEXT_en)
    de_fields = ('tar', TEXT_de) 

    examples = []
    with open(en_fname, 'r') as f_en, \
         open(de_fname, 'r') as f_de:

        for line_en, line_de in tqdm(zip(f_en, f_de), total=get_num_lines(en_fname)):
            ex = Example.fromlist([line_en, line_de], [en_fields, de_fields])
            examples.append(ex) # Examples stores tokenized sequential data


    ds_train = Dataset(
        examples=examples,
        fields=[en_fields, de_fields]
    )
    # Till now, no tokenization, no vocabulary

    start_time = time.time()
    TEXT_en.build_vocab(ds_train, min_freq=args.min_freq)
    print("EN Vocab Built. Time Taken:{}s".format(time.time() - start_time))

    start_time = time.time()
    TEXT_de.build_vocab(ds_train, min_freq=args.min_freq)
    print("DE Vocab Built. Time Taken:{}s".format(time.time() - start_time))

    sorted(examples, key=lambda x: len(x.src)) # x.src is a list of tokenized src sentence

    # [TODO] Handle Long sentences (some ignore field is present)

    idx, B = 0, args.batch_size 
    out_base_dir = os.path.join(os.path.join(args.data_dir, "tokenzied"))
    out_dir = os.path.join(os.path.join(out_base_dir, f"{split}"))
    os.makedirs(out_dir, exist_ok=True)
    file_name_idx = 0

    while idx < len(examples):
        end = min(idx + B, len(examples))
        # [TODO] Can improve batching so as to reduce padding, save some space
        batch = Batch(
            data=examples[idx:end],
            dataset=ds_train, # Most likely used to access the Fields 
            device=device
        )
        idx += B
        data_en, data_de = batch.src.numpy(), batch.tar.numpy()

        with open(os.path.join(out_dir, f"{split}_{file_name_idx:02}"), "wb") as f:
            np.savez(f, src=data_en, tar=data_de)
            file_name_idx += 1

    if split == "train":
        vocab = {
            "en" : {
                "stoi" : TEXT_en.vocab.stoi,
                "itos" : TEXT_en.vocab.itos
            },
            "de" : {
                "stoi" : TEXT_de.vocab.stoi,
                "itos" : TEXT_de.vocab.itos
            }
        }

        with open(os.path.join(out_base_dir, "vocab.pth"), "wb") as f:
            pickle.dump(vocab, f)

if __name__ == "__main__":
    for split in args.splits:
        process(split=split)
