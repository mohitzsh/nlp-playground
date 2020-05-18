from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import os
from models.advanced_models import Encoder, Attention, Decoder, Seq2Seq
import random
from typing import Tuple
from torchtext.data.metrics import bleu_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import math
import time

from tqdm import tqdm
def nop(it, *a, **k):
    return it
tqdm = nop
from tensorboardX import SummaryWriter

import argparse
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_dataset():
    SRC = Field(tokenize = "spacy",
                tokenizer_language="en",
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)

    TRG = Field(tokenize = "spacy",
                tokenizer_language="de",
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)

    train_data, val_data, test_data = Multi30k.splits(exts = ('.en', '.de'),
                                                        fields = (SRC, TRG))
    

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    if args.overfit:
        train_data.examples = train_data.examples[:args.overfit_n]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, val_iterator, test_iterator = BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size = args.batch_size,
        device = device)
    
    return train_iterator, val_iterator, test_iterator, SRC, TRG


def load_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--train-val_split-randseed", type=int, default=42)
    parser.add_argument("--train-val-split-ratio", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--min-freq-en", type=int, default=1)
    parser.add_argument("--min-freq-de", type=int, default=1)

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float,default=0.5)
    parser.add_argument("--attention-dim", type=int, default=8)
    parser.add_argument("--clip-grad", type=float, default=1)

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
    
    parser.add_argument("--overfit", action='store_true',
        help="Debug mode. Overfit to one training example and evaluate on that example.")
    parser.add_argument("--overfit-n", type=int, default=1,
        help="Number of training examples to overfit on.")

    return parser.parse_args()

args = load_args()

# Directory Name
args.exp_name = f"{args.exp_id}" \
                f"--em-dim-{args.embedding_dim}" \
                f"--h-dim-{args.hidden_dim}"

args.log_dir = os.path.join(args.log_dir,args.exp_name)

writer = SummaryWriter(args.log_dir)
global_step = 0
global_epoch = 0

# [TODO] Do this later
#args.output_log_dir = os.path.join(args.log_dir, "outputs")
#os.makedirs(args.output_log_dir, exist_ok=True)
#
## SETUP Logging of validation translations as HTML
#def setup_val_translation_logging():
#    args.val_translation_file = os.path.join(args.log_dir, "index.html")
#    css_file = os.path.join(args.log_dir, "txtstyle.css")
#    with open(args.val_translation_file, "w") as f:
#        f.write(f"<link href=\"txtstyle.css\" rel=\"stylesheet\" type=\"text/css\" />\n")
#
#    with open(css_file, "w") as f:
#        f.write("html, body { font-family: Helvetica, Arial, sans-serif; white-space: pre-wrap; }")
#
#setup_val_translation_logging()

def stringify(num_sentences, vocab):
    """Convert numericized sentences to string sentences"""
    # num_sentences is BxT tensor of numericized sentences
    # [NOTE] See, type hints are very important
    num_sentences = num_sentences.tolist()
    ignore_tokens = [vocab.stoi[t] for t in ['<sos>', '<eos>', '<pad>']]

    str_sentences = []    
    for s in num_sentences:
        # Filter s, to remote <sos>, <eos>, <pad> tokens
        s = list(
            filter(lambda x, ignore_tokens=ignore_tokens: x not in ignore_tokens , s)
        )


        # stringify
        s = list(
            map(lambda t, itos=vocab.itos: itos[t], s)
        )

        s = list(
            filter(lambda x: not x.isspace(), s)
        )

        str_sentences.append(s)

    return str_sentences

def train(model: nn.Module,
        iterator: BucketIterator,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        clip: float):

    model.train()

    global global_step
    epoch_loss = 0

    for _, batch in tqdm(enumerate(iterator), total=len(iterator.dataset)/args.batch_size):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        _output = output.clone()

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        writer.add_scalar("Loss", loss.item(),  global_step)

        global_step += 1

    return epoch_loss / len(iterator), _output

def evaluate(model: nn.Module,
            iterator: BucketIterator,
            criterion: nn.Module,
            pad_idx: int):

    model.eval()
    epoch_loss = 0
    translations = []
    references = []

    with torch.no_grad():

        for bidx, batch in tqdm(enumerate(iterator)):
            
            src = batch.src
            trg = batch.trg

            L, B = trg.shape

            output = model(src, trg, 0) #turn off teacher forcing

            _translations = output[1:].max(dim=-1)[1]

            translations.append(_translations.cpu())
            references.append(trg.cpu())

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(iterator)

    return epoch_loss, translations, references

def epoch_time(start_time: int,
            end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def compute_bleu(candidates, references, vocab, max_n=4):
    """
    candidates: TxB
    references: TxB
    """
    special_tokens = [vocab.stoi[t] for t in ['<pad>', '<sos>', '<eos>']]
    #ignore_tokens = [vocab.stoi[t] for t in [' ']]

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

    for ref in tqdm(references):
        _references.append(_clean(ref.squeeze(-1), special_tokens))
    for can in tqdm(candidates):
        _candidates.append(_clean(can.squeeze(-1), special_tokens))

    references = [] 
    candidates = []
    
    for ref in tqdm(_references):
        references.append([list(map(lambda x: vocab['itos'][x], ref))])
    
    for can in tqdm(_candidates):
        candidates.append(list(map(lambda x: vocab['itos'][x], can)))

    score = bleu_score(candidates, references, max_n=max_n)

    return score, candidates, references

def main():

    # DATA LOADING
    train_iterator, val_iterator, test_iterator, SRC, TRG = load_dataset()
    args.src_vocab_size = len(SRC.vocab.itos)
    args.tar_vocab_size = len(TRG.vocab.itos)

    ###############
    # MODEL SETUP #
    ###############

    enc = Encoder(
        input_dim=args.src_vocab_size,
        emb_dim=args.embedding_dim,
        enc_hid_dim=args.hidden_dim,
        dec_hid_dim=args.hidden_dim,
        dropout=args.dropout
    )

    attn = Attention(
        enc_hid_dim=args.hidden_dim,
        dec_hid_dim=args.hidden_dim,
        attn_dim=args.attention_dim
    )
        
    dec = Decoder(
        output_dim=args.tar_vocab_size,
        emb_dim=args.embedding_dim,
        enc_hid_dim=args.hidden_dim,
        dec_hid_dim=args.hidden_dim,
        dropout=args.dropout,
        attention=attn
    )

    model = Seq2Seq(
        encoder=enc,
        decoder=dec,
        device=device
    )

    model = model.to(device)

    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)


    model.apply(init_weights)


    print(f'The model has {model.parameter_count:,} trainable parameters')

    PAD_IDX = TRG.vocab.stoi['<pad>']

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    ############
    # TRAINING #
    ############

    for epoch in tqdm(range(args.epochs)):
        
        # Debug train and eval mode of the model
        #model.train()
        #batch = next(iter(train_iterator))

        #train_output = model(batch.src, batch.trg, 0)
        #model.eval()

        #eval_output = model(batch.src, batch.trg, 0)

        #if not (train_output == eval_output).byte().all():
        #    import ipdb; ipdb.set_trace()
        
        start_time = time.time()

        train_loss = 0
        if not args.skip_train:
            train_loss, train_output = train(model, train_iterator, optimizer, criterion, args.clip_grad)


        if args.overfit:
            val_loss, _translations, _references = evaluate(model, train_iterator, criterion, PAD_IDX)
        else:
            val_loss, _translations, _references = evaluate(model, val_iterator, criterion, PAD_IDX)

        end_time = time.time()

        translations = []
        references = []
        for t, r in zip(_translations, _references):
            translations += stringify(t.T.contiguous(), TRG.vocab)
            references += stringify(r.T.contiguous(), TRG.vocab)
        

        references = [[r] for r in references]
        
        #if args.overfit:
        #    references = list(map(lambda ex: [ex.trg] , train_iterator.dataset.examples))
        #else:
        #    references = list(map(lambda ex: [ex.trg] , val_iterator.dataset.examples))

        try:
            score = bleu_score(translations, references)
        except:
            import ipdb; ipdb.set_trace()

        #if args.overfit:
        #    print(f"EPOCH: {epoch}")
        #    for i in range(args.overfit_n):
        #        print("REF :\t"+ ' '.join(references[i][0]))
        #        print("TRNS:\t"+ ' '.join(translations[i]))

        writer.add_scalar("BLEU", score, epoch)
        writer.add_scalar("Val Loss", val_loss, epoch)

        #print(f"Epoch: {epoch+1:02} | BLEU: {score:7.3f}")

        #epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        ##print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        #print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        #print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')

        print(f"EPOCH: {epoch} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val Loss: {val_loss:.3f} | Bleu: {score:7.3f}")

if __name__ == "__main__":
    main()
