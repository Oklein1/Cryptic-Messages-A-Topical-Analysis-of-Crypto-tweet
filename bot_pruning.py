import pdb
import csv
import copy
import random
import pickle
import itertools
from text_processing import process_tweet
from sklearn.naive_bayes import MultinomialNB


DATA_CSV_LOC = 'Bitcoin_tweets.csv'
OUTFILE_LOC = 'tagged_data.txt'
PICKLE_LOC = 'mnb_pickle.pickle'
NUM_TO_TAG = 300 # number of lines to manually tag on 1 run of this file
DATA_CSV_LEN = 414548 # number of lines in data file


def predict(tokens, wordmap, padlen, clf):
    clf_tokens = copy.deepcopy(tokens)

    # Pad tokens list to length that clf expects
    while len(clf_tokens) < padlen:
        clf_tokens.append('')

    # Convert tokens to integers as done in training
    # and convert token to '' if it hasnt been seen yet
    for i in range(len(clf_tokens)):
        if clf_tokens[i] not in wordmap:
            clf_tokens[i] = ''
        clf_tokens[i] = wordmap[clf_tokens[i]]

    return bool(clf.predict([clf_tokens]))


def train():
    tokens_list = []
    tags = []
    with open(OUTFILE_LOC, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            tags.append(int(line[-1]))
            tokens_list.append(line[:-1].strip().split())

    # Pad every list of tokens to be the same length
    longest_tokens = 0
    for tokens in tokens_list:
        if longest_tokens < len(tokens):
            longest_tokens = len(tokens)
    for i in range(len(tokens_list)):
        while len(tokens_list[i]) < longest_tokens:
            tokens_list[i].append('')

    # Convert each unique token into an integer
    wordmap = {}
    current_val = 0
    for i in range(len(tokens_list)):
        tokens = tokens_list[i]
        for token in tokens:
            if token not in wordmap:
                wordmap[token] = current_val
                current_val += 1
    for i in range(len(tokens_list)):
        for j in range(len(tokens_list[i])):
            tokens_list[i][j] = wordmap[tokens_list[i][j]]

    clf = MultinomialNB()
    clf.fit(tokens_list, tags)

    # Save clf and processing parameters to a pickle object
    pickle_dict = {
        'clf': clf,
        'wordmap': wordmap,
        'padlen': longest_tokens
    }
    with open(PICKLE_LOC, 'wb') as f:
        f.write(pickle.dumps(pickle_dict))


# Gets a specific line from the CSV. MUCH faster than just iterating through csv until desired line
def get_csv_lines(csv_loc, line):
    with open(csv_loc, encoding='utf8') as f:
        yield next(itertools.islice(csv.reader(f), line, None))


def manually_tag_data(csv_loc):
    sample_lines = [random.randint(0, DATA_CSV_LEN) for _ in range(NUM_TO_TAG)]

    outfile = open(OUTFILE_LOC, 'a+', encoding='utf8')

    num_tagged = 0

    for line_num in sample_lines:
        for line in get_csv_lines(DATA_CSV_LOC, line_num):
            text = line[9]

            print('\n\nTWEET %s/%s' % (num_tagged, NUM_TO_TAG))
            print('#'*100)
            print(text)
            print('#'*100)

            tag = None
            while tag not in ('0', '1', 'x'):
                if num_tagged == 0:
                    tag = input('Is this spam? (\'0\' if no, \'1\' if yes, \'x\' to exit): ')
                else:
                    tag = input('Is this spam?: ')
                if tag == 'x':
                    return

            tokens = process_tweet(text)
            for token in tokens:
                outfile.write(token + ' ')
            outfile.write(tag + '\n')
            num_tagged += 1

    outfile.close()


if __name__ == '__main__':
    option = input('Train or tag? (1 or 2): ')
    if option == '1':
        train()
    elif option == '2':
        manually_tag_data(DATA_CSV_LOC)