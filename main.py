import pickle
from time import time
from nltk import download
from bot_pruning import predict
from text_processing import get_processed_tweets
from topic_extraction import tfidf, sorted_count
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


DATA_CSV_LOC = 'Bitcoin_tweets.csv'

# Text processing parameters
DO_CLEAN = True
NLTK_SPLIT = True
DO_DESTEM = True
DO_LEMMATIZE = True
REMOVE_SW = True

# Number of tweets to process. Set small for testing, set to -1 to do entire dataset.
MAX_TWEETS = 1000

# Write out this many of the most 'important' words from each class' TFIDF results. -1 for all
MAX_WRITE = 2000

PICKLE_LOC = 'mnb_pickle'


def nltk_download():
    download('punkt') # Used in nltk text splitting
    download('wordnet') # Used in nltk lemmatizer
    download('omw-1.4') # Used in nltk lemmatizer
    download('stopwords') # Used in nltk stopword filter


def main():
    nltk_download()

    vader = SentimentIntensityAnalyzer()
    with open(PICKLE_LOC, 'rb') as f:
        pickle_dict = pickle.loads(f.read())
    wordmap = pickle_dict['wordmap']
    padlen = pickle_dict['padlen']
    clf = pickle_dict['clf']

    num_pos = 0
    num_neu = 0
    num_neg = 0

    pos_counts = {}
    neg_counts = {}
    neu_counts = {}

    num_bot = 0

    print("PROCESSING...", end='', flush=True)

    t0 = time()
    for tokens in get_processed_tweets(DATA_CSV_LOC, do_clean=DO_CLEAN, nltk_split=NLTK_SPLIT, do_destem=DO_DESTEM, do_lemmatize=DO_LEMMATIZE, remove_sw=REMOVE_SW, max_num=MAX_TWEETS):

        if predict(tokens, wordmap, padlen, clf):
            num_bot += 1
            continue

        score = vader.polarity_scores(' '.join(tokens))['compound']

        # Keep track of the total number of each word that's occured in this class
        if score > 0.05:
            num_pos += 1
            for token in tokens:
                pos_counts[token] = pos_counts[token] + 1 if token in pos_counts else 1
        elif score < -0.05:
            num_neg += 1
            for token in tokens:
                neg_counts[token] = neg_counts[token] + 1 if token in neg_counts else 1
        else:
            num_neu += 1
            for token in tokens:
                neu_counts[token] = neu_counts[token] + 1 if token in neu_counts else 1

    results = [pos_counts, neg_counts, neu_counts]

    print("DONE. %ss" % round(time() - t0, 2))
    print("Positive Tweets: %s" % num_pos)
    print("Negative Tweets: %s" % num_neg)
    print("Neutral Tweets: %s" % num_neu)
    print("Bots pruned: %s\n" % num_bot)

    # Write summaries of each class' TFIDF results out to a file

    outfile_pos = open('tfidf_results_pos.txt', 'w', encoding='utf8')
    outfile_neg = open('tfidf_results_neg.txt', 'w', encoding='utf8')
    outfile_neu = open('tfidf_results_neu.txt', 'w', encoding='utf8')
    outfiles = [outfile_pos, outfile_neg, outfile_neu]

    num_written = 0
    file = 0
    for words_scores in sorted_count(results):
        for token, score in words_scores:
            outfiles[file].write('%s %s\n' % (token, score))
            num_written += 1
            if num_written == MAX_WRITE:
                break
        num_written = 0
        file += 1

    outfile_pos.close()
    outfile_neg.close()
    outfile_neu.close()


if __name__ == '__main__':
    main()