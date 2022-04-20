import re
import pdb
import pickle
import datetime
import numpy as np
import pandas as pd
from math import log10
from bot_pruning import predict
from text_processing import process_tweet


SPAM_THRESHOLD = 0.15       # Proportion of user's tweets that need to be marked as spam before considering the whole user spam
REPUTATION_THRESHOLD = 0.35 # Reputation below which might indicate bot. Reputation = followers / (followers + following)
TWEET_COUNT_THRESHOLD = 200 # Number of tweets above which might indicate bot.
LINK_THRESHOLD = 0.9        # Proportion of tweets which contain a url, above which might indicate bot
AGE_THRESHOLD = 2           # Years from today to year of account creation, younger than which might indicate bot
TS_STDEV_THRESHOLD = 150000 # Standard deviation of timestamps (in ms), below which might indicate bot (meaning tweets are usually close together in time)
# ENTROPY_THRESHOLD = 0.61    # Entropy of user's tweets, below which might indicate a bot (tweets have consistent content)

FAILED_CHECKS_THRESHOLD = 4 # How many of these checks need to fail to mark a user as a bot


df = pd.read_csv('Bitcoin_tweets.csv')

predictions = {}

for username, grp_idx in df.groupby('user_name').groups.items():
    try:

        tweets = df.iloc[grp_idx]

        # Years from year account was created to today
        age = datetime.date.today().year - datetime.datetime.strptime(tweets.iloc[0]['user_created'], r"%Y-%m-%d %H:%M:%S").year

        # Reputation of account, where reputation = followers / (followers + following)
        reputation = float(tweets.iloc[0]['user_followers']) / (float(tweets.iloc[0]['user_followers']) + float(tweets.iloc[0]['user_friends']))

        is_verified = tweets.iloc[0]['user_verified'] == 'True'

        # Number of times a tweet was posted from each source (e.g. Web, api...)
        source_counts = {}
        for source in tweets['source']:
            source_counts[source] = source_counts.get(source, 0) + 1
        source_counts = sorted(source_counts.items(), key=lambda item: item[1], reverse=True) # sort into list [(source, count), (source, count)...] descending

        # Get times at which each tweet was made, and the standard deviation of the times
        timestamps = []
        for date in tweets['date']:
            timestamp = datetime.datetime.strptime(date, r"%Y-%m-%d %H:%M:%S").timestamp()
            timestamps.append(timestamp)
        timestamp_std = np.std(timestamps)

        # Get proportion of tweets containing any spam flags
        spam_flags = ['project', 'giveaway', 'referral', 'opportunity', 'free btc'] #'t.co']
        spam_prop = 0
        for text in tweets:
            if any([spam_flag in text for spam_flag in spam_flags]):
                spam_prop += 1
        spam_prop = spam_prop / len(tweets)

        # Calculate entropy of user's tweets, where entropy is -1 * the summation of (the probability a token occurs in their tweets) * (log_10 of the same probability)
        # num_tokens = 0
        # tweet_token_counts = {}
        # for text in tweets['text']:
        #     tokens = process_tweet(text)
        #     for token in tokens:
        #         tweet_token_counts[token] = tweet_token_counts.get(token, 0) + 1
        #         num_tokens += 1
        # tweet_entropy = 0
        # for token in tweet_token_counts.keys():
        #     probability = tweet_token_counts[token] / num_tokens
        #     tweet_entropy += probability * log10(probability)
        # tweet_entropy = -tweet_entropy

        # Count ratio of user's tweets which contain links
        num_tweets_with_links = ['http' in text for text in tweets['text']]
        link_ratio = len(num_tweets_with_links) / len(tweets)

        # Make decision on if user is a bot or not
        # This would probably be better to do with random forest
        prediction = False
        if not is_verified:
            if spam_prop > SPAM_THRESHOLD: # or svm_spam_flag
                prediction = True
            else:
                num_failed_checks = 0
                if link_ratio > LINK_THRESHOLD:
                    num_failed_checks += 1
                if age < AGE_THRESHOLD:
                    num_failed_checks += 1
                if len(tweets) > TWEET_COUNT_THRESHOLD:
                    num_failed_checks += 1
                if reputation < REPUTATION_THRESHOLD:
                    num_failed_checks += 1
                if source_counts[0][0] not in ('Twitter Web App', 'Twitter For Android', 'Twitter for iPhone'):
                    num_failed_checks += 1
                if timestamp_std < TS_STDEV_THRESHOLD:
                    num_failed_checks += 1
                # if tweet_entropy < ENTROPY_THRESHOLD:
                #     num_failed_checks += 1
                if num_failed_checks > FAILED_CHECKS_THRESHOLD:
                    prediction = True
        predictions[username] = prediction

    except Exception as e:
        # Lots of broken lines in this dataset for some reason. e.g.
        # 137068   *Muhammad Yasir* hello stalker nice to tweet ...  2009-08-31 07:40:42            280.0          623  ...  Twitter for Android    False    NaN        NaN
        # Maybe accounts that were private when data was colleted?
        # Most appear to be spam, and almost /all/ of them only have 1-3 tweets.
        # Regardless, marking as bots so they get pruned in main().
        predictions[username] = True
        print('Error processing user "%s", setting prediction to True.' % username)

try:
    # write this out to a pickle because it takes forever to run
    with open('bot_user_predictions.pickle', 'wb') as f:
        f.write(pickle.dumps(predictions))
except Exception as e:
    pdb.set_trace()
    print(e)
