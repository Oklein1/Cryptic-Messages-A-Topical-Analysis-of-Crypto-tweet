import os
import pickle
import pandas as pd
from time import time
from nltk import download
from topic_extraction import lda, get_Kmeans_labels
from plots import plot_2d_vader_classes, plot_seaborn_kmeans
from text_processing import process_tweet, curry_text_cleaner
from bot_pruning_2 import make_bot_pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



DATA_CSV_LOC = 'data/covid19_tweets.csv'
DF_PICKLE_LOC = 'pickles/df_covid.pickle'
BOT_PICKLE_LOC = 'pickles/bot_user_predictions_covid.pickle'

WRITE_DF_PICKLE  = False  # If this is true, the pandas dataframe (with data, tokens, bot marks, vader scores, vader classes) will be pickled. This file can get very large.
FORCE_REGEN_DF   = True   # If this is true, df will be remade even if a pickle of it it already exists. LDA and outfiles will not be rewritten.
FORCE_REGEN_BOTS = True   # If this is true, bot pickle will be regenerated even if it already exists.

CLUSTERS = 6 ## PART II VARIABLE

MAX_TWEETS = 200 #10000        # Number of tweets to process. Set small for testing, set to -1 to do entire dataset. If you want to change this, make sure you aren't set to use df pickle.


# Downloads requirements for NLTK operations used in text processing
def nltk_download():
    download('punkt')       # Used in nltk text splitting
    download('wordnet')     # Used in nltk lemmatizer
    download('omw-1.4')     # Used in nltk lemmatizer
    download('stopwords')   # Used in nltk stopword filter


# Downloads requirements that don't come with pip install and creates directories main() will use
def init():
    nltk_download()
    for path in ('./results', './pickles', './data'):
        if not os.path.isdir(path):
            os.mkdir(path)


def write_tokens_lda(lda_results):

    for vader_class in lda_results:

        vader_class_to_name = {1: 'pos', -1: 'neg', 0: 'neu'}
        outfile = 'results/results_tokens_lda_%s.txt' % vader_class_to_name[vader_class]

        longest_word_len = max([max([len(word) for word, rating in topic]) for topic in lda_results[vader_class]])

        with open(outfile, 'w', encoding='utf8') as f:
            for i in range(len(lda_results[vader_class])):
                f.write('TOPIC %s\n' % i)
                for word, rating in lda_results[vader_class][i]:
                    word = word.ljust(longest_word_len, ' ')
                    rating = round(rating, 2)
                    f.write('\t%s %s%%\n' % (word, rating))
                f.write('\n')


def write_bot_tweets(df):
    with open('results/bot_tweets.txt', 'w+', encoding='utf8') as outfile:
        for bot_tweet in df[(df['is_bot'] == 1)]['text']:
            outfile.write(bot_tweet.replace('\n', '') + '\n')


vader = SentimentIntensityAnalyzer()
def get_vader_scores(tokens):
    scores = vader.polarity_scores(' '.join(tokens))
    return ((scores['pos'], scores['neg'], scores['neu'], scores['compound']))


def get_vader_class(scores):
    compound = scores[3]
    return 1 if compound > 0.05 else -1 if compound < -0.05 else 0


def ts(t):
    print("Done. %ss" % round(time() - t, 2))


def main():

    # Timestamp used to track total runtime of main()
    t0 = time()

    init()
    print()

    df_made_from_scratch = False

    if not os.path.exists(DF_PICKLE_LOC) or FORCE_REGEN_DF:

        t = time()
        print("Reading csv...", end='', flush=True)
        if MAX_TWEETS != -1:
            df = pd.read_csv(DATA_CSV_LOC, nrows=MAX_TWEETS)
        else:
            df = pd.read_csv(DATA_CSV_LOC)
        ts(t)


        if not os.path.exists(BOT_PICKLE_LOC) or FORCE_REGEN_BOTS:
            t = time()
            print("Marking bot users...", end='', flush=True)
            make_bot_pickle(silent=True)
            ts(t)

        t = time()
        print("Reading preprocessed bot users...", end='', flush=True)
        with open(BOT_PICKLE_LOC, 'rb') as f:
            bot_user_predictions = pickle.loads(f.read())
        ts(t)

        t = time()
        print("Tokenizing tweets...", end='', flush=True)
        df['tokens'] = df['text'].apply(lambda text: process_tweet(text))
        ts(t)

        t = time()
        print("Marking bots...", end='', flush=True)
        df['is_bot'] = df['user_name'].apply(lambda username: bot_user_predictions.get(username, False))
        ts(t)

        t = time()
        print("Getting VADER scores...", end='', flush=True)
        df['vader'] = df['tokens'].apply(lambda tokens: get_vader_scores(tokens))
        ts(t)

        t = time()
        print("Classifying VADER scores...", end='', flush=True)
        df['class'] = df['vader'].apply(lambda scores: get_vader_class(scores))
        ts(t)
        


        df_made_from_scratch = True

    else:
        t = time()
        print("Reading dataframe pickle...", end='', flush=True)
        df = pd.read_pickle(DF_PICKLE_LOC) #ACHTUNG HIER#
        ts(t)

    # df_made_from_scratch check is so pickle isn't read in, then immediately written again
    if WRITE_DF_PICKLE and df_made_from_scratch:
        t = time()
        print("Saving dataframe pickle...", end='', flush=True)
        df.to_pickle(DF_PICKLE_LOC)
        ts(t)
        

        

    t = time()
    print("Running LDA...", end='', flush=True)
    lda_results = lda(df)
    ts(t)

    t = time()
    print("Writing results...", end='', flush=True)
    write_tokens_lda(lda_results)
    write_bot_tweets(df)
    ts(t)

    print('\n'+'#'*50+'\n')
    humans = df[~(df['is_bot'] == 1)]
    print('Positive Tweets:', len(humans[(humans['class'] == 1)]))
    print('Negative Tweets:', len(humans[(humans['class'] == -1)]))
    print('Neutral Tweets:', len(humans[(humans['class'] == 0)]))
    print('Bot tweets:', len(df[(df['is_bot'] == 1)]))
    ts(t0) # print total runtime timestamp
    print('\n'+'#'*50+'\n')

    #plot_2d_vader_classes(df)
    
    ############
    # PART II: #
    ############
    
    humans['Postive_score'] = df['vader'].apply(lambda x: x[0])
    humans['Negative_score'] = df['vader'].apply(lambda x: x[1])

    data = humans[(humans.Postive_score != 0) & (humans.Negative_score != 0)] #filter out
    data = data[[ "tokens","is_bot"]] #filter out
    
    data["KMeans_label"] = get_Kmeans_labels(data, clusters=CLUSTERS) # NEED AS GLOBAL FOR LATER STUFF
    
    # data["KMeans_label"] = kmeans_fit.labels_ #MAY NOT NEED
    #plot_seaborn_kmeans(data, kmeans=kmeans_fit, clusters=CLUSTERS) 
    
    data["tokens"] = data["tokens"].apply(curry_text_cleaner) #I did it on text column
    
    
    t = time()
    print("Running LDA on Clusters...", end='', flush=True)
    lda_cluster_results = lda(data)
    ts(t)
    print(lda_cluster_results)

    # t = time()
    # print("Writing results...", end='', flush=True)
    # write_tokens_lda(lda_cluster_results)
    # ts(t)

if __name__ == '__main__':
    main()