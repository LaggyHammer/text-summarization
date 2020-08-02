# coding: utf-8
# =====================================================================
#  Filename:    extraction_summary.py
#
#  py Ver:      python 3.6 or later
#
#  Description: Performs extractive Summarization on a given text file
#
#  Usage: python extraction_summary.py --file sample_text.py --embeddings glove.6B.100d.txt --length 5
#
#  Note: Requires sklearn and nltk
#
#  Author: Ankit Saxena (ankch24@gmail.com)
# =====================================================================

import argparse
import ntpath
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import time
import nltk
nltk.download('stopwords')
nltk.download('punkt')


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', type=str, required=True,
                    help='path to text file to be summarized')
    ap.add_argument('-e', '--embeddings', type=str,
                    help='path to embeddings text file')
    ap.add_argument('-l', '--length', type=int, default=5,
                    help='length of the summary')
    arguments = vars(ap.parse_args())

    return arguments


def path_leaf(path):
    """
    Gets file name from a given path
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def ingest_data(path):
    """
    Reads data from a given path
    :param path: path to file containing text to be summarized
    :return: text to be summarized
    """
    f = open(path, 'r', encoding='utf8')
    text = f.read()
    f.close()

    return text


def get_embeddings(embedding_path):
    """
    Reads GloVe embeddings from a text file
    :param embedding_path: path to the word embeddings
    :return: dictionary containing words and their respective embeddings
    """
    word_embeddings = {}
    f = open(embedding_path, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefficients
    f.close()

    return word_embeddings


def remove_stopwords(sentence):
    """
    Removes stopwords from a given sentence
    """
    sen_new = " ".join([i for i in sentence if i not in stopwords.words('english')])
    return sen_new


def text_preprocessing(text, length):
    """
    Text preprocessing function
    :param length: length of the summary
    :param text: text to be pre-processed
    :return: list with original sentences & list with processed sentences
    """
    # separate out sentences
    sentences = sent_tokenize(text)
    # check if the text has at least 5 elements
    if len(sentences) < length:
        return False, False
    # remove every character except alphabets
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    # turn all characters lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    # remove stopwords from sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    return sentences, clean_sentences


def text_vectorization(sentences, embeddings):
    """
    Calculates vectors for a given list of sentences
    :param sentences: sentences to create the vectors for
    :param embeddings: embeddings to create the vectors with
    :return: list of sentence vectors
    """
    sentence_vectors = []
    for i in sentences:
        # get embeddings for each word in a sentence & sum them up
        # divide the sum with the length of the sentence
        if len(i) != 0:
            vector = sum([embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            vector = np.zeros((100,))
        sentence_vectors.append(vector)

    return sentence_vectors


def rank_text(vectors):
    """
    Rank the sentence vectors based on their similarity
    :param vectors: list of vectors to be ranked
    :return: dictionary containing the scores for each vector
    """
    # initialize similarity matrix
    similarity_matrix = np.zeros([len(vectors), len(vectors)])
    # fill the similarity matrix with cosine similarity scores
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(vectors[i].reshape(1, 100),
                                                            vectors[j].reshape(1, 100))[0, 0]

    # create a network graph from the matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    # use the pagerank algorithm to get scores for each sentence
    scores = nx.pagerank(nx_graph)

    return scores


def summarize(sentences, scores, length):
    """
    Summarizes the text by extracting the top ranked sentences
    :param sentences: list of sentences to summarize
    :param scores: scores for each sentence
    :param length: number of sentences in the summary
    :return: summarized text
    """
    # order sentence indices according to their rank
    ordered_indices = [index[0] for index in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    # extract the top sentences
    top_indices = sorted(ordered_indices)[:length]

    # append to summary
    summary = ""
    for index in top_indices:
        summary = summary + sentences[index] + " "

    return summary


def write_summary(text, title):
    """
    Writes the summary to a text file
    """
    with open('summary_' + title, 'w') as f:
        f.write(text)


def main(text_path, embedding_path, sum_length, show):

    print('[INFO] Ingesting Text...')
    text = ingest_data(text_path)
    orig_sentences, sentences = text_preprocessing(text, sum_length)

    # handle summary length mismatches
    if not orig_sentences:
        print('[ERROR] Text is shorter than summary length. Writing entire text file.')
        write_summary(text, title=path_leaf(text_path))
        return text

    print('[INFO] Loading Embeddings...')
    start = time.time()
    embeddings = get_embeddings(embedding_path)
    end = time.time()
    print(f'[INFO] Loaded embeddings in {round(end - start, 2)} seconds.')
    print('[INFO] Vectoring Text...')
    vectors = text_vectorization(sentences, embeddings)
    print('[INFO] Ranking Text...')
    scores = rank_text(vectors)
    print('[INFO] Summarizing...')
    summary = summarize(orig_sentences, scores, sum_length)
    write_summary(summary, title=path_leaf(text_path))

    if show:
        print(summary)

    return summary


if __name__ == '__main__':

    args = get_arguments()

    summary_length = args['length']

    # minimum summary length requirement
    if args['length'] < 5:
        print(f'[WARNING] Summary length less than 5. Setting length to 5')
        summary_length = 5

    main(text_path=args['file'], embedding_path=args['embeddings'], sum_length=summary_length, show=True)
