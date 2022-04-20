import os
import csv
import pickle
import pandas as pd
from time import time
from nltk import download
from string import punctuation
from plots import plot_2d_vader_classes
from bot_pruning import predict, PICKLE_LOC
from text_processing import process_tweet
from topic_extraction import sorted_count
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


DATA_CSV_LOC = 'Bitcoin_tweets.csv'

# Text processing parameters
DO_CLEAN = True
NLTK_SPLIT = True
DO_DESTEM = True
DO_LEMMATIZE = True
REMOVE_SW = True

# Number of tweets to process. Set small for testing, set to -1 to do entire dataset.
MAX_TWEETS = 10000

# Write out this many of the most 'important' words from each class' TFIDF results. -1 for all
MAX_WRITE = 2000


def nltk_download():
    download('punkt')       # Used in nltk text splitting
    download('wordnet')     # Used in nltk lemmatizer
    download('omw-1.4')     # Used in nltk lemmatizer
    download('stopwords')   # Used in nltk stopword filter


def init():
    nltk_download()
    if not os.path.isdir('./results'):
        os.mkdir('./results')


def is_good_token(token):
    return token not in punctuation


def write_tokens(df):
    file_names = ['results/results_tokens_pos.txt', 'results/results_tokens_neg.txt', 'results/results_tokens_neu.txt', 'results/results_tokens_bot.txt']
    sorted_counts = sorted_count(df)
    for i in range(len(file_names)):
        with open(file_names[i], 'w+', encoding='utf8') as outfile:
            tokens_written = 0
            for token, count in sorted_counts[i]:
                if is_good_token(token):
                    outfile.write('%s: %s\n' % (token, count))
                    tokens_written += 1
                    if tokens_written == MAX_WRITE:
                        break


def write_bot_tweets(df):
    with open('results/bot_tweets.txt', 'w+', encoding='utf8') as outfile:
        for bot_tweet in df[(df['is_bot'] == 1)]['text']:
            outfile.write(bot_tweet.replace('\n', '') + '\n')


def main():

    init()

    df = pd.DataFrame({
        'text': [],
        'tokens': [],
        'is_bot': [],
        'vader': [],
        'class': []
    })

    vader = SentimentIntensityAnalyzer()

    with open('bot_user_predictions.pickle', 'rb') as f:
        bot_user_predictions = pickle.loads(f.read())

    with open(DATA_CSV_LOC, 'r', encoding='utf8') as f:
        num_processed = 0
        reader = csv.DictReader(f)
        next(reader) # discard first csv row
        for line in reader:
            try:
                user = line['user_name']
                text = str(line['text'])
                tokens = process_tweet(text, do_clean=DO_CLEAN, nltk_split=NLTK_SPLIT, do_destem=DO_DESTEM, do_lemmatize=DO_LEMMATIZE, remove_sw=REMOVE_SW)

                num_processed += 1
                if (num_processed == MAX_TWEETS):
                    break

                is_bot = bot_user_predictions.get(user, False)
                scores = vader.polarity_scores(' '.join(tokens))

                vader_class = 0
                if scores['compound'] > 0.05:
                    vader_class = 1
                elif scores['compound'] < -0.05:
                    vader_class = -1

                row = pd.DataFrame({
                    'text': [text],
                    'tokens': [tokens],
                    'is_bot': [is_bot],
                    'vader': [(scores['pos'], scores['neg'], scores['neu'], scores['compound'])],
                    'class': [vader_class]
                })
                df = pd.concat([df, row], ignore_index=True)

            except:
                print('ERROR PROCESSING LINE:', line)

    humans = df[~(df['is_bot'] == 1)]
    print('Positive Tweets:', len(humans[(humans['class'] == 1)]))
    print('Negative Tweets:', len(humans[(humans['class'] == -1)]))
    print('Neutral Tweets:', len(humans[(humans['class'] == 0)]))
    print('Bot tweets:', len(df[(df['is_bot'] == 1)]))

    write_tokens(df)
    write_bot_tweets(df)

    plot_2d_vader_classes(df)


if __name__ == '__main__':
    main()