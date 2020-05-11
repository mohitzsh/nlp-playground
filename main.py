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
    num_sentences = num_sentences.tolist()
    ignore_tokens = [vocab['stoi'][t] for t in [INIT_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN]]

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

def compute_bleu(candidates, references, vocab, max_n=4):

    special_tokens = [vocab['de']['stoi'][t] for t in ['<pad>', '<sos>', '<eos>']]
    ignore_tokens = [vocab['de']['stoi'][t] for t in [' ']]

    def _clean(x, special_tokens):
        """Remove the padding tokens"""
        ret = []
        for t in list(x.numpy()):
            if t in ignore_tokens:
                # ignore whitespace in the translation
                continue
            if t not in special_tokens:
                ret.append(t)
                continue
            
            if t == vocab['de']['stoi']['<eos>']:
                break
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
        references.append([list(map(lambda x: vocab['de']['itos'][x], ref))])
    
    for can in _candidates:
        candidates.append(list(map(lambda x: vocab['de']['itos'][x], can)))

    return bleu_score(candidates, references, max_n=max_n)

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

    #TEXT_en = Field(
    #    sequential=True,
    #    use_vocab=True,
    #    init_token=INIT_TOKEN,
    #    eos_token=EOS_TOKEN,
    #    tokenize="spacy",
    #    tokenizer_language="en_core_web_sm",
    #    lower=True
    #)

    #TEXT_de = Field(
    #    sequential=True,
    #    use_vocab=True,
    #    init_token=INIT_TOKEN,
    #    eos_token=EOS_TOKEN,
    #    tokenize="spacy",
    #    tokenizer_language="de_core_news_sm",
    #    lower=True
    #)

    #en_fields = ('src_en', TEXT_en)
    #de_fields = ('tar_de', TEXT_de) 

    ## Create Train/Val dataset from the English file   
    #train_examples = []
    #with open(train_en_file, 'r') as f_en, \
    #     open(train_de_file, 'r') as f_de:

    #     for line_en, line_de in tqdm(zip(f_en, f_de)):
    #         ex = Example.fromlist([line_en, line_de], [en_fields, de_fields])
    #         train_examples.append(ex)

    ## [TODO] handle sort_key stuff 
    #ds_train = Dataset(examples=train_examples, fields=[en_fields, de_fields])

    #ds_train, ds_val = ds_train.split(split_ratio=args.train_val_split_ratio, random_state=train_val_split_randstate)

    #import pdb; pdb.set_trace()
    #start_time = time.time()
    #TEXT_en.build_vocab(ds_train, min_freq=args.min_freq_en)
    #print("EN Vocab Built. Time Taken:{}s".format(time.time() - start_time))

    #start_time = time.time()
    #TEXT_de.build_vocab(ds_train, min_freq=args.min_freq_de)
    #print("DE Vocab Built. Time Taken:{}s".format(time.time() - start_time))

    #import pdb; pdb.set_trace()

    #en_vocab_size = len(TEXT_en.vocab.itos)
    #de_vocab_size = len(TEXT_de.vocab.itos)

    #train_iter = Iterator(
    #    dataset=ds_train,
    #    batch_size=args.batch_size,
    #    train=True,
    #    device=device,
    #)

    ################
    # SETUP MODELS #
    ################
    encoder = LSTMEncoder(
        vocab_size=en_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )

    decoder = LSTMDecoder(
        vocab_size=de_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )

    encoder, decoder = encoder.to(device), decoder.to(device)

    H_e = H_d = args.hidden_dim
    
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = optim.Adam(params, lr = args.lr)

    it = 0

    for epoch in range(args.epochs):
        encoder.train(); decoder.train()
        for batch_idx, (src, tar) in enumerate(train_dataloader):
            if args.skip_train:
                break
            encoder.train(); decoder.train() 

            it += 1      

            src, tar = src.T, tar.T

            assert tar.shape[0] < args.max_seq_len, "Training data doesn't satisfy the max_seq_len constraint"

            B = src.size()[1]

            src_enc = encoder(src) # TxBx2*H

            eos_pos = torch.nonzero(src.T ==  vocab['en']['stoi'][EOS_TOKEN])[:, -1]
            eos_pos = eos_pos.unsqueeze(0).unsqueeze(-1).long()

            eos_pos = eos_pos.expand(-1, -1, src_enc.shape[-1])
            context = torch.gather(src_enc, 0, eos_pos)

            context = context.view(2, -1, args.hidden_dim)
            #mask = (src ==  vocab['en']['stoi'][EOS_TOKEN]).unsqueeze(-1) # TxBx1

            #context = src_enc.masked_select(mask).reshape(2, B, H_e) # LSTM expects num_layers*directions as the first dimension

            h, c = context, torch.zeros_like(context)
            x = torch.ones((1, B), dtype=torch.long) * vocab['de']['stoi'][INIT_TOKEN]

            loss_mask = torch.ones(B, dtype=torch.long)
            cumul_loss_mask = torch.ones_like(loss_mask)
            h, c, x, loss_mask, cumul_loss_mask = h.to(device), c.to(device), x.to(device), loss_mask.to(device), cumul_loss_mask.to(device)

            seq_len = tar.size()[0]

            loss = torch.zeros(B, dtype=torch.float).to(device)
            out_list = []

            outputs = torch.zeros(seq_len-1, B, len(vocab['de']['itos']), dtype=torch.float).to(device)
            for t in range(seq_len-1):
                # out is prob distribution over decoded sequences
                out, h, c = decoder(x, h, c) 
                outputs[t] = out

                top1 = out.argmax(dim=1).unsqueeze(0) # 1xB
                out_list.append(top1.cpu()[0])

                y = tar[t + 1] # Ground Truth
                #loss_t = nn.NLLLoss(reduction='none')(out, y)
                #loss += loss_mask.float() * loss_t
                #loss_t = nn.NLLLoss(reduction='none',ignore_index=vocab['de']['stoi'][EOS_TOKEN] )(out, y)
                #loss += loss_t

                x = y.unsqueeze(0) if random.uniform(0, 1) < args.teacher_forcing_prob else top1

                # use ignore class attribute of loss functions
                #loss_mask = loss_mask & (y != vocab['de']['stoi'][EOS_TOKEN])

                #cumul_loss_mask += loss_mask

            outputs = outputs.view(-1, len(vocab['de']['itos']))
            gt = tar[1:, :].reshape(-1)

            loss = nn.NLLLoss(ignore_index=vocab['de']['stoi'][EOS_TOKEN])(outputs, gt)

            decoded_trans = torch.stack(out_list, dim=1) # BxT
            pretty_translations = stringify(decoded_trans, vocab['de'])
            #if epoch == 1:
            #    import pdb; pdb.set_trace()
            #loss = loss / cumul_loss_mask
            #loss = torch.mean(loss)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
            opt.step()

            del src; del tar; del src_enc; del context; del h; del c; del x; del loss_mask; del out; del y
            torch.cuda.empty_cache()
            print("[ITER]:{}\t[Loss]:{:.4f}".format(it,loss.item()))
            writer.add_scalar("Loss", loss.item(), it)
        

        ##############
        # VALIDATION #
        #############

        encoder.eval(), decoder.eval()

        trans_list = []
        tar_list = []
        src_list = []
        for src, tar in dev_dataloader:

            src, tar = src.T, tar.T # Only works for 2D tensor

            tar_list.append(tar.cpu()) 
            src_list.append(src.cpu())

            B = src.size()[1]
            with torch.no_grad():
                src_enc = encoder(src)

            eos_pos = torch.nonzero(src.T ==  vocab['en']['stoi'][EOS_TOKEN])[:, -1]
            eos_pos = eos_pos.unsqueeze(0).unsqueeze(-1).long()

            eos_pos = eos_pos.expand(-1, -1, src_enc.shape[-1])
            context = torch.gather(src_enc, 0, eos_pos)

            context = context.view(2, -1, args.hidden_dim)

            h, c = context, torch.zeros_like(context)
            trans_b = [] 

            x = torch.ones((1, B), dtype=torch.long) * vocab['de']['stoi'][INIT_TOKEN]

            h, c, x = h.to(device), c.to(device), x.to(device)

            trans_b.append(x.cpu()[0])

            for _ in range(args.max_seq_len):
                out, h, c = decoder(x, h, c) 
                x = out.argmax(dim=1).unsqueeze(0) # 1xB
                trans_b.append(x.cpu()[0])
            
            x = torch.ones((1, B), dtype=torch.long) * vocab['de']['stoi'][EOS_TOKEN]
            trans_b.append(x[0])

            trans_b = torch.stack(trans_b,dim=1).T
            trans_list.append(trans_b)
        
        sources = torch.cat(src_list, dim=1)
        candidates = torch.cat(trans_list, dim=1)
        references = torch.cat(tar_list, dim=1) 

        bleu = compute_bleu(candidates, references, vocab)
        writer.add_scalar("Bleu", bleu, epoch)
        print(f"BLEU: {bleu}")

        ##########################
        # VISUALIZE TRANSLATIONS #
        ##########################

        pretty_source = stringify(sources.T, vocab['en'])
        pretty_candidates = stringify(candidates.T, vocab['de'])
        pretty_references = stringify(references.T, vocab['de'])

        translation_string = ""
        separator = "-"*100
        for idx, (src, can, ref) in enumerate(zip(pretty_source, pretty_candidates, pretty_references)):
            translation_string += f"[{epoch}.{idx}][SRC]\t{' '.join(src)}\n"
            translation_string += f"[{epoch}.{idx}][REF]\t{' '.join(ref)}\n[{epoch}.{idx}][CAN]\t{' '.join(can)}\n"
            translation_string += (separator + "\n")

        writer.add_text("translations", translation_string, epoch)

        with open(args.val_translation_file, "a") as f:
            f.write(f"{translation_string}")



if __name__ == "__main__":

    main()
