# from lib2to3.pytree import _Results
from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

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

def topic_extractor_VADERClusters(df, groupbyColumn):
    """Used to extract topics when cluster is 3:
    All positives, all neutrals, all negatives."""
    vectorizer = CountVectorizer(max_df=0.95, min_df=3, max_features=5000)
    lda = decomposition.LatentDirichletAllocation(n_components=LDA_NUM_TOPICS, random_state=42)
    
    results = {}

    humans = df[~(df['is_bot'] == 1)] #ACHTUN: set i here for cluster loop with condition
    for vader_class, grp_idx in humans.groupby(groupbyColumn).groups.items(): # CHANGE TO CLUSTERS

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


def display_topics(model, features, cluster_number, no_top_words=10):
    storage = []
    print("\nCluster %02d" % cluster_number,file=open(f"./results/KMeansCluster{cluster_number}_Topics.txt", "a"))
    for topic, word_vector in enumerate(model.components_):
        total = word_vector.sum()
        largest = word_vector.argsort()[::-1]
        print("\nTopic %02d" % topic, file=open(f"./results/KMeansCluster{cluster_number}_Topics.txt", "a")) #
        for i in range(0, no_top_words):
            print(" %s (%2.2f)" % (features[largest[i]], word_vector[largest[i]]*100.0/total),file=open(f"./results/KMeansCluster{cluster_number}_Topics.txt", "a"))


def topic_extractor_KmeanClusters(df):
    for i in df['KMeans_label'].unique():
        kmeans_group = df[df['KMeans_label']== i][["tokens","KMeans_label","is_bot"]]
            
        #Vectorize
        vectorizer = CountVectorizer(max_df=0.95, min_df=3, max_features=5000)
        tf_vectors = vectorizer.fit_transform(kmeans_group['tokens'])
        tf_feature_names = vectorizer.get_feature_names()
    
        # LDA topic modeling
        num_of_topics = 10 # Set the number of topics you want to extract
        lda = decomposition.LatentDirichletAllocation(n_components=num_of_topics, random_state=42)
        lda.fit_transform(tf_vectors)
        
            #topics
        display_topics(lda, tf_feature_names, cluster_number=i)
        print(f"Cluster {i} is complete. Moving on...")

    return "Topics Extracted and stored in files. Apologies for the side-effect."
    


def lda(df):
    if not "KMeans_label" in df.columns:
        return topic_extractor_VADERClusters(df, "class")
    
    else:
        return topic_extractor_KmeanClusters(df)
            



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


    ############
    # PART II: #
    ############

def get_Kmeans(df, clusters):
    """PART II EXTRACTOR"""
    kmeans = KMeans(n_clusters=clusters, #CHOOSE K clusters 
                    init='k-means++', 
                    random_state=0).fit(df[["Postive_score","Negative_score"]])
    return kmeans


