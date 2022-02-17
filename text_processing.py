import csv
import re


STOP_WORDS_LOC = 'stop_words.txt'


def filter_punctuation(str):
    return re.sub(r'[^\w\s]', '', str)


def filter_hashtags(str):
    return re.sub(r'#\S+(\s|$)', '', str) # '#' followed by characters, then a space or end of line


def filter_stop_words(str, stop_words):
    return ' '.join([word for word in str.split() if word not in stop_words])


def filter_short_words(str, min_len=3):
    return ' '.join([word for word in str.split() if len(word) >= min_len])


def filter_links(str):
    return re.sub(r'http\S+(\s|$)', '', str) # 'http' followed by characters, then a space or end of line


def filter_numbers(str):
    return re.sub(r'\d', '', str)


def filter_all(str, stop_words=None):
    str = str.lower()
    str = filter_hashtags(str)
    str = filter_punctuation(str)
    str = filter_links(str)
    str = filter_numbers(str)
    str = filter_short_words(str)
    if stop_words:
        str = filter_stop_words(str, stop_words)
    return str


def read_stop_words(file_loc=STOP_WORDS_LOC):
    with open('stop_words.txt', 'r') as f:
        stop_words = f.read().strip().split('\n')
        stop_words = [filter_all(word) for word in stop_words]
    return stop_words


def test_process_csv(csv_loc):
    stop_words = read_stop_words()
    with open(csv_loc, 'r') as f:
        reader = csv.DictReader(f)
        next(reader) # discard first csv row
        for line in reader:
            text = filter_all(line['text'], stop_words)
            text = tuple(text.split())
            if len(text):
                print(text)


if __name__ == '__main__':
    test_process_csv('Bitcoin_tweets.csv')