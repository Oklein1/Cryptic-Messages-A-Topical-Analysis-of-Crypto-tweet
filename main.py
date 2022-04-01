import pdb
from time import time
from nltk import download, RegexpParser
from text_processing import get_processed_tweets
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TESTING = False # trigger a breakpoint every time a new tweet is read

DATA_CSV_LOC = 'Bitcoin_tweets.csv'

# Text processing parameters
DO_CLEAN = True
NLTK_SPLIT = True
DO_DESTEM = True
DO_LEMMATIZE = True
REMOVE_SW = True

# Number of tweets to process. Set small for testing, set to -1 to do entire dataset.
MAX_TWEETS = 1000


def nltk_download():
    download('punkt') # Used in nltk text splitting
    download('wordnet') # Used in nltk lemmatizer
    download('omw-1.4') # Used in nltk lemmatizer
    download('stopwords') # Used in nltk stopword filter


def main():
    print()
    nltk_download()

    vader = SentimentIntensityAnalyzer()

    num_pos = 0
    num_neu = 0
    num_neg = 0

    print("PROCESSING...", end='', flush=True)

    t0 = time()
    t = t0
    for tokens in get_processed_tweets(DATA_CSV_LOC, do_clean=DO_CLEAN, nltk_split=NLTK_SPLIT, do_destem=DO_DESTEM, do_lemmatize=DO_LEMMATIZE, remove_sw=REMOVE_SW, max_num=MAX_TWEETS):
        tweet = ' '.join(tokens)
        score = vader.polarity_scores(tweet)['compound']

        if score > 0.05:
            num_pos += 1
        elif score < -0.05:
            num_neg += 1
        else:
            num_neu += 1

        # Print a . every 5 seconds
        if time() - t > 5:
            print('.', end='', flush=True)
            t = time()

        if TESTING:
            pdb.set_trace()

    print("DONE. %ss\n" % round(time() - t0, 2))

    print("CLASS COUNTS")
    print("Positive Tweets: %s" % num_pos)
    print("Negative Tweets: %s" % num_neg)
    print("Neutral Tweets: %s" % num_neu)

    print()

if __name__ == '__main__':
    main()