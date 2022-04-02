from math import log10

DISCARD_PUNC = True
DISCARD_EMOJI = False


# https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
def tfidf(pos_neg_neu_word_counts):

    total_words = [0, 0, 0] # total number of words in each class
    word_counts_all = {} # tracks count of each word encountered across all classes
    for i in (0, 1, 2):
        word_counts = pos_neg_neu_word_counts[i]
        for word in word_counts.keys():
            total_words[i] += word_counts[word]
            if word in word_counts_all:
                word_counts_all[word] += word_counts[word]
            else:
                word_counts_all[word] = word_counts[word]

    for i in (0, 1, 2):
        word_scores = {}
        word_counts = pos_neg_neu_word_counts[i]
        for word in word_counts.keys():
            tf = word_counts[word] / total_words[i] # how many times a word occurs in a doc / how many words there are total in that doc
            df = word_counts_all[word] # number of times word occurs across all documents.
            idf = log10(4 / df) + 1 # 4 = number of docs + 1. https://programminghistorian.org/en/lessons/analyzing-documents-with-tfidf
            tfidf = tf * idf
            word_scores[word] = tfidf
        yield sorted(word_scores.items(), key=lambda item: item[1], reverse=True)


def sorted_count(pos_neg_neu_word_counts):
    for word_counts in pos_neg_neu_word_counts:
        yield sorted(word_counts.items(), key=lambda item: item[1], reverse=True)