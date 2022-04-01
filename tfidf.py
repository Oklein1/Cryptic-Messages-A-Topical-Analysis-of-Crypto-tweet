from math import log10

# https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
# NOTE: in this case, document is tweets in one sentiment class
def tfidf(word_counts, num_tweets):
    word_scores = {}

    # total number of words in entire set
    total_words = sum([word_counts[key] for key in word_counts.keys()])

    for word in word_counts.keys():
        tf = word_counts[word] / total_words # how many times a word occurs / how many words there are
        df = word_counts[word] # number of times word occurs across all documents.
        #idf = log10(num_tweets / (df + 1))
        idf = num_tweets / (df + 1)
        tfidf = tf * idf
        word_scores[word] = tfidf

    return sorted(word_scores.items(), key=lambda item: item[1], reverse=True)
