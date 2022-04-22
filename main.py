import os
import pickle
import pandas as pd
from time import time
from nltk import download
from topic_extraction import lda
from plots import plot_2d_vader_classes
from text_processing import process_tweet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


DATA_CSV_LOC = 'data/covid19_tweets.csv'
BOT_PICKLE_LOC = 'pickles/bot_user_predictions_covid.pickle'


MAX_TWEETS = 10000 # Number of tweets to process. Set small for testing, set to -1 to do entire dataset.
MAX_WRITE = 2000 # Write out this many of the most 'important' words from each class' TFIDF results. -1 for all


# Downloads requirements for NLTK operations used in text processing
def nltk_download():
    download('punkt')       # Used in nltk text splitting
    download('wordnet')     # Used in nltk lemmatizer
    download('omw-1.4')     # Used in nltk lemmatizer
    download('stopwords')   # Used in nltk stopword filter


# Downloads requirements that don't come with pip install and creates directories main() will use
def init():
    nltk_download()
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    if not os.path.isdir('./pickles'):
        os.mkdir('./pickles')


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

    t = time()
    print("Reading csv...", end='', flush=True)
    if MAX_TWEETS != -1:
        df = pd.read_csv(DATA_CSV_LOC, nrows=MAX_TWEETS)
    else:
        df = pd.read_csv(DATA_CSV_LOC)
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

    plot_2d_vader_classes(df)


if __name__ == '__main__':
    main()