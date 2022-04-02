import pdb
import csv
import random
import itertools
from text_processing import process_tweet


DATA_CSV_LOC = 'Bitcoin_tweets.csv'
OUTFILE_LOC = 'tagged_data.txt'
NUM_TO_TAG = 300 # number of lines to manually tag on 1 run of this file
DATA_CSV_LEN = 414548 # number of lines in data file


# Gets a specific line from the CSV. MUCH faster than just iterating through csv until desired line
def get_csv_lines(csv_loc, lines):
    with open(csv_loc, encoding='utf8') as f:
        for line in lines:
            yield next(itertools.islice(csv.reader(f), line, None))


def manually_tag_data(csv_loc):
    sample_lines = [random.randint(0, DATA_CSV_LEN) for _ in range(NUM_TO_TAG)]

    outfile = open(OUTFILE_LOC, 'a+', encoding='utf8')

    num_tagged = 0

    for line in get_csv_lines(DATA_CSV_LOC, sample_lines):
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
    manually_tag_data(DATA_CSV_LOC)