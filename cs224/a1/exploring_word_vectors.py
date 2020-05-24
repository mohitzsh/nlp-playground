# All Import Statements Defined Here
# Note: Do not add to this list.
# ----------------

import sys
assert sys.version_info[0]==3
assert sys.version_info[1] >= 5

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import nltk
nltk.download('reuters')
from nltk.corpus import reuters
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)
# ----------------
from itertools import chain

def read_corpus(category="crude"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)

    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    
    # ------------------
    # Write your implementation here.
    
    corpus = set(list(chain(*corpus)))
    
    num_corpus_words = len(corpus)
    corpus_words = sorted(list(corpus))
    # ------------------

    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "<START> All that glitters is not gold <END>" with window size of 4,
              "All" will co-occur with "<START>", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (a symmetric numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}
    
    # ------------------
    # Write your implementation here.

    # [NOTE] : Index is not going to be consistent between two different runs of the program.
    word2Ind = {w: idx for idx, w in enumerate(words)}
    
    M = np.zeros((num_words, num_words), dtype=int)

    # Naive solution
    # ------------------
    def update_cooccurance_doc(doc, M):

        for i, word_i in enumerate(doc):
            max_idx = min(len(doc) - 1 , i + window_size) + 1
            for j in range(i +1, max_idx):
                x = word2Ind[word_i]
                y = word2Ind[doc[j]]

                # Going to be symmetric matrix
                M[x,y] += 1
                M[y,x] += 1
    
    for doc in corpus:
        update_cooccurance_doc(doc, M)

    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`

    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ------------------
    # Write your implementation here.
    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)

    M_reduced = svd.fit_transform(M)
    # ------------------

    print("Done.")
    return M_reduced

def plot_embeddings(M_reduced, word2Ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , 2)): matrix of 2-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    # ------------------
    # Write your implementation here.
    idx_to_plot = [word2Ind[w] for w in words]

    M_reduced_to_plot = M_reduced[idx_to_plot]

    fig, ax = plt.subplots()

    ax.scatter(x=M_reduced_to_plot[:,0], y=M_reduced_to_plot[:,1]) 

    for w in words:
        idx = word2Ind[w]
        ax.annotate(w, (M_reduced[idx,0], M_reduced[idx,1]))
    # ------------------

    return fig

def load_embedding_model():
    """ Load GloVe Vectors
        Return:
            wv_from_bin: All 400000 embeddings, each lengh 200
    """
    import gensim.downloader as api
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(wv_from_bin.vocab.keys()))
    return wv_from_bin

def get_matrix_of_vectors(wv_from_bin, required_words=['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']):
    """ Put the GloVe vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 400000 GloVe vectors loaded from file
        Return:
            M: numpy matrix shape (num words, 200) containing the vectors
            word2Ind: dictionary mapping each word to its row number in M
    """
    import random
    words = list(wv_from_bin.vocab.keys())
    print("Shuffling words ...")
    random.seed(224)
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        if w in words:
            continue
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind

def main():
    reuters_corpus = read_corpus()
    pprint.pprint(reuters_corpus[:3], compact=True, width=100)

    # ---------------------
    # Run this sanity check
    # Note that this not an exhaustive check for correctness.
    # ---------------------

    # Define toy corpus
    test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
    test_corpus_words, num_corpus_words = distinct_words(test_corpus)

    # Correct answers
    ans_test_corpus_words = sorted([START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
    ans_num_corpus_words = len(ans_test_corpus_words)

    # Test correct number of words
    assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)

    # Test correct words
    assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))

    # Print Success
    print ("-" * 80)
    print("Passed All Tests!")
    print ("-" * 80)
    # ---------------------
    # Run this sanity check
    # Note that this is not an exhaustive check for correctness.
    # ---------------------

    # Define toy corpus and get student's co-occurrence matrix
    test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
    M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)

    # Correct M and word2Ind
    M_test_ans = np.array( 
        [[0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,],
        [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,],
        [0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,],
        [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,],
        [1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,],
        [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,],
        [0., 0., 1., 0., 1., 1., 0., 0., 0., 1.,],
        [1., 0., 0., 1., 1., 0., 0., 0., 1., 0.,]]
    )
    ans_test_corpus_words = sorted([START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
    word2Ind_ans = dict(zip(ans_test_corpus_words, range(len(ans_test_corpus_words))))

    # Test correct word2Ind
    assert (word2Ind_ans == word2Ind_test), "Your word2Ind is incorrect:\nCorrect: {}\nYours: {}".format(word2Ind_ans, word2Ind_test)

    # Test correct M shape
    assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape, M_test_ans.shape)

    # Test correct M values
    for w1 in word2Ind_ans.keys():
        idx1 = word2Ind_ans[w1]
        for w2 in word2Ind_ans.keys():
            idx2 = word2Ind_ans[w2]
            student = M_test[idx1, idx2]
            correct = M_test_ans[idx1, idx2]
            if student != correct:
                print("Correct M:")
                print(M_test_ans)
                print("Your M: ")
                print(M_test)
                raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))

    # Print Success
    print ("-" * 80)
    print("Passed All Tests!")
    print ("-" * 80)

    # ---------------------
    # Run this sanity check
    # Note that this is not an exhaustive check for correctness 
    # In fact we only check that your M_reduced has the right dimensions.
    # ---------------------

    # Define toy corpus and run student code
    test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
    M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
    M_test_reduced = reduce_to_k_dim(M_test, k=2)

    # Test proper dimensions
    assert (M_test_reduced.shape[0] == 10), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 10)
    assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

    # Print Success
    print ("-" * 80)
    print("Passed All Tests!")
    print ("-" * 80)

    # ---------------------
    # Run this sanity check
    # Note that this is not an exhaustive check for correctness.
    # The plot produced should look like the "test solution plot" depicted below. 
    # ---------------------

    print ("-" * 80)
    print ("Outputted Plot:")

    #M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
    #word2Ind_plot_test = {'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
    #words = ['test1', 'test2', 'test3', 'test4', 'test5']
    #plot_embeddings(M_reduced_plot_test, word2Ind_plot_test, words)

    #print ("-" * 80)


    # -----------------------------
    # Run This Cell to Produce Your Plot
    # ------------------------------
    reuters_corpus = read_corpus()
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)

    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

    words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']

    fig = plot_embeddings(M_normalized, word2Ind_co_occurrence, words)

    fig.savefig("embed_co_occurance.png")
    print("Word embeddings form Co-occurance matrix plotted.")


    # -----------------------------------
    # Run Cell to Load Word Vectors
    # Note: This will take several minutes
    # -----------------------------------
    wv_from_bin = load_embedding_model()

    # -----------------------------------------------------------------
    # Run Cell to Reduce 200-Dimensional Word Embeddings to k Dimensions
    # Note: This should be quick to run
    # -----------------------------------------------------------------
    M, word2Ind = get_matrix_of_vectors(wv_from_bin)
    M_reduced = reduce_to_k_dim(M, k=2)

    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced, axis=1)
    M_reduced_normalized = M_reduced / M_lengths[:, np.newaxis] # broadcasting

    words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
    fig = plot_embeddings(M_reduced_normalized, word2Ind, words)

    fig.savefig("embed_glove.png")
    print("Word embeddings form glove embedding plotted..")


if __name__ == "__main__":
    main()