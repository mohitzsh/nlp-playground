import os
import argparse
import random
import logging
import time
from tqdm import tqdm
import numpy as np
import pickle
import glob
import random

import spacy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchtext.data import Field, Dataset, Example, Iterator
from torchtext.data.metrics import bleu_score

from models.basic_models import LSTMEncoder, LSTMDecoder
from models.advanced_models import Encoder, Decoder, Attention, Seq2Seq

from tensorboardX import SummaryWriter

INIT_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
MODEL_CKPT_EXT = ".pth" # {epoch}.{MODEL_CHECKPOINT_EXT}
LATEST_CKPT_NAME = "latest" 

def stringify(num_sentences, vocab):
    """Convert numericized sentences to string sentences"""
    # num_sentences is BxT tensor of numericized sentences
    # [NOTE] See, type hints are very important
    num_sentences = num_sentences.tolist()
    ignore_tokens = [vocab['stoi'][t] for t in [INIT_TOKEN, EOS_TOKEN, PAD_TOKEN]]

    # [TODO] Can you remove this loop?
    str_sentences = []    
    for s in num_sentences:
        # Filter s, to remote <sos>, <eos>, <pad> tokens
        s = list(
            filter(lambda x, ignore_tokens=ignore_tokens: x not in ignore_tokens , s)
        )
        # stringify
        s = list(
            map(lambda t, itos=vocab['itos']: itos[t], s)
        )

        str_sentences.append(s)

    return str_sentences


# [NOTE] This is where type hints would be helpful, I don't remember 
# what datatypes should have been used for candidates
def compute_bleu(candidates, references, vocab, max_n=4):
    """
    candidates: T x B
    references: T x B
    """
    candidates = candidates.T
    references = references.T
    special_tokens = [vocab['stoi'][t] for t in ['<pad>', '<sos>', '<eos>']]
    ignore_tokens = [vocab['stoi'][t] for t in [' ']]

    def _clean(x, special_tokens):
        """Remove the padding tokens"""
        ret = []
        for t in list(x.numpy()):
            try:
                if t in ignore_tokens:
                    # ignore whitespace in the translation
                    continue
                if t not in special_tokens:
                    ret.append(t)
                    continue
                
                if t == vocab['stoi']['<eos>']:
                    break
            except:
                import ipdb; ipdb.set_trace()
        return ret 
    

    N = references.shape[-1]  
    references = references.chunk(N, dim=-1)
    candidates = candidates.chunk(N, dim=-1)

    _references = []
    _candidates = []

    for ref in references:
        _references.append(_clean(ref.squeeze(-1), special_tokens))
    for can in candidates:
        _candidates.append(_clean(can.squeeze(-1), special_tokens))

    references = [] 
    candidates = []
    
    for ref in _references:
        references.append([list(map(lambda x: vocab['itos'][x], ref))])
    
    for can in _candidates:
        candidates.append(list(map(lambda x: vocab['itos'][x], can)))

    score = bleu_score(candidates, references, max_n=max_n)

    return score, candidates, references

def load_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--train-val_split-randseed", type=int, default=42)
    parser.add_argument("--train-val-split-ratio", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=int, default=1e-3)

    parser.add_argument("--min-freq-en", type=int, default=1)
    parser.add_argument("--min-freq-de", type=int, default=1)

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float,default=0.5)
    parser.add_argument("--attention_dim", type=int, default=8)

    parser.add_argument("--max-seq-len", type=int, default=50)

    parser.add_argument("--log-dir", type=str, default="logs" )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--exp-id", type=str, default="main",
        help="Prefix for experiment related files.")

    parser.add_argument("--skip-train", action='store_true',
        help="Skip Training to reach the evaluation code. Used for debugging")

    parser.add_argument("--max-n", type=int, default=4,
        help="max-n argument for bleu_score computation")
    
    parser.add_argument("--teacher-forcing-prob", type=float, default=0,
        help="Rate of teacher forcing.")

    return parser.parse_args()

args = load_args()

# Get rand state to be used for splitting the training file
random.seed(args.train_val_split_randseed)
train_val_split_randstate = random.getstate()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Directory Name
args.exp_name = f"{args.exp_id}" \
                f"--em-dim-{args.embedding_dim}" \
                f"--h-dim-{args.hidden_dim}"

args.log_dir = os.path.join(args.log_dir,args.exp_name)

writer = SummaryWriter(args.log_dir)

args.output_log_dir = os.path.join(args.log_dir, "outputs")
os.makedirs(args.output_log_dir, exist_ok=True)



# SETUP Logging of validation translations as HTML

def setup_val_translation_logging():
    args.val_translation_file = os.path.join(args.log_dir, "index.html")
    css_file = os.path.join(args.log_dir, "txtstyle.css")
    with open(args.val_translation_file, "w") as f:
        f.write(f"<link href=\"txtstyle.css\" rel=\"stylesheet\" type=\"text/css\" />\n")

    with open(css_file, "w") as f:
        f.write("html, body { font-family: Helvetica, Arial, sans-serif; white-space: pre-wrap; }")

setup_val_translation_logging()
class WMT14Dataset(Dataset):
    """
    Load tokenized WMT14 Datsaet stored as processed numpy array, split across multiple files
    """

    def __init__(self, data_dir, vocab, split='train', max_seq_len=100, device=None):
        self.data_dir = data_dir 
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.split = split
        assert max_seq_len > 0, f"Invalid maximum sequence length(={max_seq_len})."\
                                f"Has to be positive" 
        self._load_data()
        self.device = device if device is not None else torch.device("cpu")

    def _load_data(self): 
        # [NOTE] NEVER DO self.src = self.tar = []

        # [TODO] This function is horrible. Group by size and then shuffle among similar length examples
        self.src = []
        self.tar = []

        pattern = self.data_dir + f"/{self.split}_[0-9][0-9]"
        paths = glob.glob(pattern)
        sorted(paths, key=lambda path: int(path.split('_')[-1]))

        for path in tqdm(paths):
            data = np.load(path)
            self.src.append(data['src'])
            self.tar.append(data['tar'])

        # Find the batch with maximum sequence length
        eos_mask_src = [(x == self.vocab['en']['stoi']['<eos>']).astype(int) for x in self.src] 
        eos_mask_tar = [(x == self.vocab['de']['stoi']['<eos>']).astype(int) for x in self.tar]

        eos_pos_src = [np.nonzero(x.T)[-1] for x in eos_mask_src]
        eos_pos_tar = [np.nonzero(x.T)[-1] for x in eos_mask_tar]

        # Seq is valid is it's shorter than self.max_seq_len (excluding <sos>, and <eos> tokens)
        # And is non-empty (len > 2, <sos><eos> is empty)
        valid_len_mask_src = [np.logical_and(x - 1 <= self.max_seq_len, x - 1 > 0) for x in eos_pos_src] # Batches to keep
        valid_len_mask_tar = [np.logical_and(x - 1 <= self.max_seq_len, x - 1 > 0) for x in eos_pos_tar]

        valid_indices = [np.nonzero(np.logical_and(x, y).astype(int))[0] for x,y in zip(valid_len_mask_src, valid_len_mask_tar)] 

        self.src = [np.take(x,indices,axis=1)[:self.max_seq_len+2, :] for indices, x in zip(valid_indices, self.src)]
        self.tar = [np.take(x,indices,axis=1)[:self.max_seq_len+2, :] for indices, x in zip(valid_indices, self.tar)]

        #eos_pos_src = [np.take(x, i, axis=0) for i, x in zip(valid_indices, eos_pos_src)]
        #eos_pos_tar = [np.take(x, i, axis=0) for i, x in zip(valid_indices, eos_pos_tar)]

        #self.src = [b[:self.max_seq_len+1, :] for b in self.src]
        #self.tar = [b[:self.max_seq_len+1, :] for b in self.tar]

        self.src = np.hstack(self.src)
        self.tar = np.hstack(self.tar)

        print(f"Filtered data size: {self.src.shape[1]}")

        assert self.src.shape[1] == self.tar.shape[1], \
                f"Data should have equal source-target pairs." \
                f"Src:{self.src.shape[1]}, Tar:{self.tar.shape[1]}"


    def __len__(self):
        return self.src.shape[1]

    def __getitem__(self, idx):
        return torch.from_numpy(self.src[:, idx]).to(self.device), \
             torch.from_numpy(self.tar[:, idx]).to(self.device)

def main():
    #######################
    # SETUP TRAINING DATA #
    #######################

    with open(os.path.join(args.data_dir, "vocab.pth"), "rb") as f:
        vocab = pickle.load(f) 

    en_vocab_size, de_vocab_size = len(vocab['en']['itos']), len(vocab['de']['itos'])
    train_dir = os.path.join(args.data_dir, "train")
    dev_dir = os.path.join(args.data_dir, "dev")

    train_dataset = WMT14Dataset(
        data_dir=train_dir,
        vocab=vocab,
        split='train',
        max_seq_len=args.max_seq_len,
        device=device
    )

    dev_dataset = WMT14Dataset(
        data_dir=dev_dir,
        vocab=vocab,
        split='dev',
        max_seq_len=args.max_seq_len,
        device=device
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )


    ################
    # SETUP MODELS #
    ################

    enc = Encoder(
        input_dim=en_vocab_size,
        emb_dim=args.embedding_dim,
        enc_hid_dim=args.hidden_dim,
        dec_hid_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    attn = Attention(
        enc_hid_dim=args.hidden_dim,
        dec_hid_dim=args.hidden_dim,
        attn_dim=args.attention_dim # This is not dot-product attention
    ).to(device)

    dec = Decoder(
        output_dim=de_vocab_size,
        emb_dim=args.embedding_dim,
        enc_hid_dim=args.hidden_dim,
        dec_hid_dim=args.hidden_dim,
        dropout=args.dropout,
        attention=attn
    ).to(device)

    model = Seq2Seq(
        encoder=enc,
        decoder=dec,
        device=device
    ).to(device)


    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)


    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), )


    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['de']['stoi'][INIT_TOKEN])
    

    it = 0

    for epoch in range(args.epochs):
        model.train()

        for batch_idx, (src, tar) in enumerate(train_dataloader):
            if args.skip_train:
                break

            it += 1      

            src, tar = src.T, tar.T

            assert tar.shape[0] < args.max_seq_len, "Training data doesn't satisfy the max_seq_len constraint"

            # use model for training, encoder and decoder separatelty for validation
            output = model(src, tar)  # T x B x V_{tar}

            # merging contiguous dimensions is safe, use 
            # See following for how to safely merge non-contiguous dimensions
            #  https://medium.com/mcgill-artificial-intelligence-review/the-dangers-of-reshaping-and-other-fun-mistakes-ive-learnt-from-pytorch-b6a5bdc1c275

            output = output[1:].view(-1, output.shape[-1]) # output[0] is zero and not used
            tar = tar[1:].contiguous().view(-1) # tar[0] is start of sentence token 

            loss = criterion(output, tar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            print(f"ITER:\t{it:4}\tLOSS:{loss.item():>10.04}")
        
        score, translations, references = evaluate(model, dev_dataloader, vocab)

        print(f"{'#'*50}")
        print(f"{' '*20+'EVALUATION'+' '*20}")
        print(f"BATCH:\t{epoch}\tSCORE:{score}")
        print(f"{'#'*50}")

        # Dump the translations into an html file

        create_evaluation_logs(references, translations, epoch)


def create_evaluation_logs(references, translations, epoch):
    with open(os.path.join(args.output_log_dir, f"eval_{epoch}.csv"), "w") as writer:
        
        writer.write("Reference,Translation\n")
        for ref, trans in zip(references, translations):
            writer.write(f"{' '.join(ref[0])},{' '.join(trans)}\n")

def evaluate(model, dataloader, vocab):
    model.eval()

    translations = []
    references = []
    for src, tar in dataloader:
        # This would break on a cpu
        references.append(tar.cpu())
        src = src.T
        tar = tar.T # Just using for <sos> tokens


        encoder_outputs, hidden = model.encoder(src)

        output = tar[0, :]

        b_translations = []
        for t in range(1, args.max_seq_len):

            output, hidden = model.decoder(output, hidden, encoder_outputs)

            top1 = output.max(1)[1]

            output = top1

            b_translations.append(top1.cpu())
        
        translations.append(torch.stack(b_translations, axis=-1))

    translations = torch.cat(translations, dim=0) # B x max_seq_len
    references = torch.cat(references, dim=0) # B x max_seq_len
    
    score, candidates, references = compute_bleu(translations, references, vocab['de'])
    
    return score, candidates, references

if __name__ == "__main__":

    main()
