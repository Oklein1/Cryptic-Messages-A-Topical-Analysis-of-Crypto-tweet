from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer

LDA_NUM_TOPICS = 10
NUM_TOP_WORDS = 10


# Returns results of performing LDA on df['tokens']
# Results will be of form:
# {
#     '1': [ # vader class, e.g. 1=positive
#         [(word0, rating0), (word1, rating1), ...], # topic 0
#         [(word0, rating0), (word1, rating1), ...], # topic 1
#         ...
#     ],
#     ...
# }
# LDA classifies each tweet as belonging to 1 of LDA_NUM_TOPICS possible topics,
# and the 'rating' of each word in the result represents the probability of a word occuring in a document (a tweet)
# given that that tweet is part of a given topic.
def lda(df):

    vectorizer = CountVectorizer(max_df=0.95, min_df=3, max_features=5000)
    lda = decomposition.LatentDirichletAllocation(n_components=LDA_NUM_TOPICS, random_state=42)

    results = {}

    humans = df[~(df['is_bot'] == 1)]
    for vader_class, grp_idx in humans.groupby('class').groups.items():

        vectors = vectorizer.fit_transform(df.iloc[grp_idx]['tokens'].apply(lambda tokens: ' '.join(tokens)))
        feature_names = vectorizer.get_feature_names()
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


# Simply counts the number of times each token occurs in bot messages, positive messages, negative messages, and neutral messages.
# Works, but doesn't produce very useful results.
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