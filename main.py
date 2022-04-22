import os
import csv
import pdb
import pickle
import pandas as pd
from time import time
from nltk import download
from string import punctuation
from plots import plot_2d_vader_classes
from text_processing import process_tweet
from topic_extraction import sorted_count
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


DATA_CSV_LOC = 'covid19_tweets.csv'
BOT_PICKLE_LOC = 'bot_user_predictions_covid.pickle'


MAX_TWEETS = 10000 # Number of tweets to process. Set small for testing, set to -1 to do entire dataset.
MAX_WRITE = 2000 # Write out this many of the most 'important' words from each class' TFIDF results. -1 for all


def nltk_download():
    download('punkt')       # Used in nltk text splitting
    download('wordnet')     # Used in nltk lemmatizer
    download('omw-1.4')     # Used in nltk lemmatizer
    download('stopwords')   # Used in nltk stopword filter


def init():
    nltk_download()
    if not os.path.isdir('./results'):
        os.mkdir('./results')


def write_tokens(df):
    file_names = ['results/results_tokens_pos.txt', 'results/results_tokens_neg.txt', 'results/results_tokens_neu.txt', 'results/results_tokens_bot.txt']
    sorted_counts = sorted_count(df)
    for i in range(len(file_names)):
        with open(file_names[i], 'w+', encoding='utf8') as outfile:
            tokens_written = 0
            for token, count in sorted_counts[i]:
                if token not in punctuation:
                    outfile.write('%s: %s\n' % (token, count))
                    tokens_written += 1
                    if tokens_written == MAX_WRITE:
                        break


def write_bot_tweets(df):
    with open('results/bot_tweets.txt', 'w+', encoding='utf8') as outfile:
        for bot_tweet in df[(df['is_bot'] == 1)]['text']:
            outfile.write(bot_tweet.replace('\n', '') + '\n')


vader = SentimentIntensityAnalyzer()
def get_vader_scores(tokens):
    scores = vader.polarity_scores(' '.join(tokens))
    return ((scores['pos'], scores['neg'], scores['neu'], scores['compound']))


def get_vader_class(scores):
    if scores[-1] > 0.05:
        return 1
    if scores[-1] < -0.05:
        return -1
    return 0


def ts(t):
    print("Done. %ss" % round(time() - t, 2))


def main():

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

    print('\n'+'#'*50+'\n')
    humans = df[~(df['is_bot'] == 1)]
    print('Positive Tweets:', len(humans[(humans['class'] == 1)]))
    print('Negative Tweets:', len(humans[(humans['class'] == -1)]))
    print('Neutral Tweets:', len(humans[(humans['class'] == 0)]))
    print('Bot tweets:', len(df[(df['is_bot'] == 1)]))
    print('\n'+'#'*50+'\n')

    t = time()
    print("Writing outfiles...", end='', flush=True)
    write_tokens(df)
    write_bot_tweets(df)
    ts(t)

    plot_2d_vader_classes(df)

    print('\n'+'#'*50+'\n')
    ts(t0)


if __name__ == '__main__':
    main()