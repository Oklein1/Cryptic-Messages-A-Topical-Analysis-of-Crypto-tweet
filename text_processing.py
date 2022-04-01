import csv
import re


STOP_WORDS_LOC = 'stop_words.txt'


def read_stop_words(file_loc=STOP_WORDS_LOC):
    with open(STOP_WORDS_LOC, 'r') as f:
        return f.read().strip().split('\n')


# removes unicode ellipse character from end of truncated tweets
def filter_ellipse(str):
    return re.sub(r'\u2026', '', str)

def filter_extra_whitespace(str):
    return re.sub(r'\s+', ' ', str)

# removes all non-ASCII characters, like emojis
def filter_nonascii(str):
    return re.sub(r'[^\x00-\x7F]+', '', str)

def filter_numbers(str):
    return re.sub(r'\d', '', str)

def filter_punctuation(str):
    return re.sub(r'[^\w\s]', '', str) # [^\w\s] = not alphanumeric/emoji and not a space


def filter_links(str):
    return re.sub(r'(http|www)\S+(\s|$)', '', str) # 'http' or 'www' followed by characters, then a space or end of line

# Removes numbers and any symbols/punctuation surrounding them
def filter_numbers_symbols(str):
    return re.sub(r'([^\w\s]+)?\d+([^\w\s]+)?', '', str) # numbers possibly with symbols adjacent


def filter_short_words(str, min_len=3):
    return ' '.join([word for word in str.split() if len(word) >= min_len])

def filter_stop_words(str, stop_words):
    return ' '.join([word for word in str.split() if word not in stop_words])


def filter_hashtags(str):
    return re.sub(r'#\S+(\s|$)', '', str) # '#' followed by characters, then a space or end of line

def filter_mentions(str):
    return re.sub(r'@\S+(?=(\s|$))', '', str) # '@' followed by characters, then a space or end of line

def filter_tweet_syntax(str):
    str = filter_mentions(str)
    str = filter_hashtags(str)
    if str[0:3] == 'RT ':
        str = str[3:]
    return str


# Cleans tweet string as is done in:
# "A Complete VADER-Based Sentiment Analysis of Bitcoin (BTC) Tweets during the Era of COVID-19"
def clean(str):
    str = filter_ellipse(str)
    str = filter_numbers_symbols(str)
    str = filter_links(str)
    str = filter_tweet_syntax(str)
    str = filter_extra_whitespace(str)
    return str


def tokenize(str):
    stop_words = read_stop_words()

    # Chars will be considered emoticons if they are both
    # in this group and adjacent to something else in this group
    emoticon_chars = '$=@&_*#>:\'\</{})]|%;~-,([+^"'

    groups = []

    group = ''
    group_type = None

    for i in range(len(str)):

        char = str[i]
        char_type = None

        # First, decide on char's type

        if char and char.strip() == '': # Char contains only whitespace characters
            char_type = 'space'
        else:
            if re.search(r'\w', char) or re.search(r'[^\x00-\x7F]', char): # Char is alphanumeric or an emoji
                char_type = 'alphanumeric'
            else: # Char is a punctuation character
                if char in emoticon_chars:
                    # Char could be an emoticon, but only if its next to another emoticon character
                    if (i > 0 and str[i-1] in emoticon_chars) or (i < len(str)-1 and str[i+1] in emoticon_chars):
                        char_type = 'emoticon'
                    else:
                        char_type = 'punctuation'
                else:
                    char_type = 'punctuation'

        # Now, decide if char should be added to current group, or a new group made instead

        if group != '': # there is an existing group
            if char_type == group_type:
                group += char
            else:
                # End this group and start a new one
                if group not in stop_words:
                    groups.append(group)
                if char_type == 'space':
                    group = ''
                    group_type = None
                else:
                    group = char
                    group_type = char_type
        else:
            group = char
            group_type = char_type

    return groups


def get_processed_tweets(csv_loc, stop_words_loc=STOP_WORDS_LOC):
    pass


if __name__ == '__main__':
    get_processed_tweets('Bitcoin_tweets.csv')