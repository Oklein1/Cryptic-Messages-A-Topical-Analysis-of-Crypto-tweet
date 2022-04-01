import pdb
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


def nltk_download():
    download('punkt') # Used in nltk text splitting
    download('wordnet') # Used in nltk lemmatizer
    download('omw-1.4') # Used in nltk lemmatizer
    download('stopwords') # Used in nltk stopword filter


def main():
    nltk_download()

    vader = SentimentIntensityAnalyzer()

    num_pos = 0
    num_neu = 0
    num_neg = 0

    for tokens in get_processed_tweets(DATA_CSV_LOC, do_clean=DO_CLEAN, nltk_split=NLTK_SPLIT, do_destem=DO_DESTEM, do_lemmatize=DO_LEMMATIZE, remove_sw=REMOVE_SW):
        tweet = ' '.join(tokens)
        score = vader.polarity_scores(tweet)['compound']

        if score > 0.05:
            num_pos += 1
        elif score < -0.05:
            num_neg += 1
        else:
            num_neu += 1

        if TESTING:
            pdb.set_trace()

if __name__ == '__main__':
    main()