# Implementation of GloVe : Global Vector for Word Representationa [Link](https://nlp.stanford.edu/pubs/glove.pdf)

## 1. Notes
- X_ij : # word j appears in the context of word i
- X_i : \sum_{k} X_ik
- P_ij : X_ij / X_i Probability of word j appearing in the context of word i.
- Argues that ratio of probabilities is a good statistic (read section 3). My guess is that during modeling, in a context window with center word w_i, you can maximize P(w_j|w_i) / P(w_j|w_r), w_j being the observed word in the context and w_r being a random word (you can choose w_r to be a non-context word also but if the vocabulary is large, random sampling might be good enough). In terms of the notation used in the paper, you are interested in modeling P_ik/P_jk.
- After some hand-wavy arguments, we arrive at Eq. 7 which serves as our model for parameter estimation. Here it is:  
    w_i^T \tilde{w_k} + b_i + \tilde{b}_j = log(X_ik)  
    The whys of the modeling choice might become more apparent once you read log-linear and log-bilinear models used in NLP.

- Parameter estimation is done using weighted least-squares. The objective function is  
    J = \sum_{i, j=1}^{|V|} f(X_ij)(w_i^T \tilde{w_k} + b_i + \tilde{b}_j - log(X_ik))^2  
    The choice of f(X_ij) is explained in the papers nicely.


## Experiments

### 1. Dataset


## TODOs
- [ ] Read section 3.1 to understand relationship with other models.
- [ ] Understand log-linear and log-bilinear models.

## Further Readings
- log-linear and log-bilinear models in NLP.
    - http://www.cs.utoronto.ca/~hinton/csc2535/notes/hlbl.pdf
    - https://piotrmirowski.files.wordpress.com/2014/06/piotrmirowski_2014_wordembeddings.pdf
    - http://www.cs.columbia.edu/~mcollins/loglinear.pdf
