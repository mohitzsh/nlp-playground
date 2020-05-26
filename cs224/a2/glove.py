"""
Author: Mohit Sharma
Email:  mohshar@microsoft.com

Implementation of Glove: Global Vectors for Word Representation. 

First step is to compare the performance against the vectors produced in demo.sh for the toy dataset.
Reference: https://github.com/stanfordnlp/GloVe/blob/master/demo.sh
"""
import argparse
import collections

##################
# LOAD ARGUMENTS #
##################

def load_args():
    parser = argparse.ArgumentParser()

    # Data 
    parser.add_argument("--corpus-path", type=str, default="data/text8",
        help="File to use for corpus")
    
    parser.add_argument("--vocab-size", type=int, default=50,
        help="Number of most frequent words to keep in vocabulary.")
    
    # Algorithm
    parser.add_argument("--window-size", type=int, default=5,
        help="Window size to define the context.")
    
    
    return parser.parse_args()
args = load_args()

class Glove:
    def __init__(
        self,
        corpus_path,
        vocab_size=50
    ):
        self.corpus_path = corpus_path
        self.vocab_size = vocab_size

    def build_vocab(self):
        if not hasattr(self, 'vocab') or self.vocab is None:
                corpus = self._read_corpus()
                # Use an ordered dict here
                self.freq = dict(collections.Counter(corpus).most_common(self.vocab_size))

                self.itostr = list(self.freq.keys())
                self.strtoi = {word: idx for idx, word in enumerate(self.itostr)}

                self.vocab = {
                    'strtoi' : self.strtoi,
                    'itos' : self.itostr,
                    'freq' : self.freq
                }

    def _read_corpus(self):
        """
        Returns a list of words in the corpus. If the corpus has multiple documents,
        concatenate the list of words for each documents and return a flattened one-
        dimensional list.
        """

        with open(self.corpus_path, 'r') as f:
            line = f.read()
        
        # in the demo corpus, all words are whitespace separated
        return line.split()

    def build_co_occurance_matrix(self):

        if not hasattr(self, 'co_matrix') or self.co_matrix is None:
            pass
    
    def train(self):
        pass

    def get_embedding(self, word):
        pass

def main():
    """
    1. Construct vocabulary
    2. Construct Word Co-occurance matrix.
    3. Use Pytorch's auto-differentiation to solve Glove's least-square formualtion to learn word-vectors (parameters). 
    """

    glove = Glove(corpus_path=args.corpus_path)

    glove.build_vocab()
    glove.build_co_occurance_matrix()
    glove.train()

if __name__ == "__main__":
    main()
