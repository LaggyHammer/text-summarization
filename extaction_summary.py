import argparse
import ntpath
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import time


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
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def ingest_data(path):
    f = open(path, 'r', encoding='utf8')
    text = f.read()
    f.close()

    return text


def get_embeddings(embedding_path):
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
    sen_new = " ".join([i for i in sentence if i not in stopwords.words('english')])
    return sen_new


def text_preprocessing(text):
    sentences = sent_tokenize(text)
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    return sentences, clean_sentences


def text_vectorization(sentences, embeddings):
    sentence_vectors = []
    for i in sentences:

        if len(i) != 0:
            vector = sum([embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            vector = np.zeros((100,))
        sentence_vectors.append(vector)

    return sentence_vectors


def rank_text(vectors):
    similarity_matrix = np.zeros([len(vectors), len(vectors)])
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(vectors[i].reshape(1, 100),
                                                            vectors[j].reshape(1, 100))[0, 0]

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    return scores


def summarize(sentences, scores, length):
    ordered_indices = [index[0] for index in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    top_indices = sorted(ordered_indices)[:length]

    summary = ""
    for index in top_indices:
        summary = summary + sentences[index] + " "

    return summary


def main(text_path, embedding_path, sum_length, show):

    print('[INFO] Ingesting Text...')
    text = ingest_data(text_path)
    orig_sentences, sentences = text_preprocessing(text)
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


def write_summary(text, title):
    with open('summary_' + title, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    args = get_arguments()

    main(text_path=args['file'], embedding_path=args['embeddings'], sum_length=args['length'], show=True)
