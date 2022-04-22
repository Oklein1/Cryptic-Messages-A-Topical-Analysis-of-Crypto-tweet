from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer

LDA_NUM_TOPICS = 10
NUM_TOP_WORDS = 10

# # https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
# def tfidf(pos_neg_neu_word_counts):
#
#     total_words = [0, 0, 0] # total number of words in each class
#     word_counts_all = {} # tracks count of each word encountered across all classes
#     for i in (0, 1, 2):
#         word_counts = pos_neg_neu_word_counts[i]
#         for word in word_counts.keys():
#             total_words[i] += word_counts[word]
#             if word in word_counts_all:
#                 word_counts_all[word] += word_counts[word]
#             else:
#                 word_counts_all[word] = word_counts[word]
#
#     for i in (0, 1, 2):
#         word_scores = {}
#         word_counts = pos_neg_neu_word_counts[i]
#         for word in word_counts.keys():
#             tf = word_counts[word] / total_words[i] # how many times a word occurs in a doc / how many words there are total in that doc
#             df = word_counts_all[word] # number of times word occurs across all documents.
#             idf = log10(4 / df) + 1 # 4 = number of docs + 1. https://programminghistorian.org/en/lessons/analyzing-documents-with-tfidf
#             tfidf = tf * idf
#             word_scores[word] = tfidf
#         yield sorted(word_scores.items(), key=lambda item: item[1], reverse=True)
#
#
# def sorted_count(pos_neg_neu_word_counts):
#     for word_counts in pos_neg_neu_word_counts:
#         yield sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

def lda(df):

    vectorizer = CountVectorizer(max_df=0.95, min_df=3, max_features=5000)
    lda = decomposition.LatentDirichletAllocation(n_components=LDA_NUM_TOPICS, random_state=42)

    # Results will be of form:
    # {
    #     '1': [ # vader class, e.g. 1=positive
    #         [(word0, rating0), (word1, rating1), ...], # topic 0
    #         [(word0, rating0), (word1, rating1), ...], # topic 1
    #         ...
    #     ],
    #     ...
    # }
    results = {}

    humans = df[~(df['is_bot'] == 1)]
    for vader_class, grp_idx in humans.groupby('class').groups.items():

        vectors = vectorizer.fit_transform(df.iloc[grp_idx]['tokens'].apply(lambda tokens: ' '.join(tokens)))
        feature_names = vectorizer.get_feature_names_out()
        lda.fit_transform(vectors)

        topics = [] # list of each topic's words and their ratings in this sentiment class
        for _, word_vector in enumerate(lda.components_):
            total = word_vector.sum()
            largest = word_vector.argsort()[::-1]
            words = [] # each word and its rating in this topic
            for i in range(0, NUM_TOP_WORDS):
                words.append((feature_names[largest[i]], word_vector[largest[i]]*100.0/total))
            topics.append(words)
        results[vader_class] = topics

    return results

def sorted_count(df):

    tokens_pos, tokens_neg, tokens_neu, tokens_bot = {}, {}, {}, {}
    bots = df[(df['is_bot'] == 1)]
    humans = df[~(df['is_bot'] == 1)]

    # Count occurences of each token in each class
    for tokens in bots['tokens']:
        tokens_bot.update(count_tokens(tokens))
    for tokens in humans[(humans['class'] == 1)]['tokens']:
        tokens_pos.update(count_tokens(tokens))
    for tokens in humans[(humans['class'] == -1)]['tokens']:
        tokens_neg.update(count_tokens(tokens))
    for tokens in humans[(humans['class'] == 0)]['tokens']:
        tokens_neu.update(count_tokens(tokens))

    # Sort each count dictionary
    result = [tokens_pos, tokens_neg, tokens_neu, tokens_bot]
    result = [sorted(result[i].items(), key=lambda item: item[1], reverse=True) for i in range(len(result))]

    return result

def count_tokens(tokens):
    result = {}
    for token in tokens:
        result[token] = result.get(token, 0) + 1
    return result