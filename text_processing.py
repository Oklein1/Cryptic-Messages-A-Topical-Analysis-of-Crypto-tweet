import csv
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


STOP_WORDS_LOC = 'stop_words.txt'

CLEAN_PUNC = False
CLEAN_EMOJI = False


def read_stop_words_file(file_loc=STOP_WORDS_LOC):
    with open(STOP_WORDS_LOC, 'r') as f:
        return f.read().strip().split('\n')


def read_stop_words_nltk():
    return set(stopwords.words('english'))


#################################################################
#################### BASIC CHARACTER FILTERS ####################
#################################################################

# removes unicode ellipse character from end of truncated tweets
def filter_ellipse(str):
    return re.sub(r'\u2026', '', str)

def filter_extra_whitespace(str):
    return re.sub(r'\s+', ' ', str) # Multiple (+) whitespace (\s) characters

# removes all non-ASCII characters, like emojis
def filter_nonascii(str):
    return re.sub(r'[^\x00-\x7F]+', '', str) # Characters not (^) between (-) ASCII code 00 (\x00) and 127 (\x7F)

def filter_numbers(str):
    return re.sub(r'\d', '', str) # Any number (\d)

def filter_punctuation(str):
    return re.sub(r'[^\w\s]', '', str) # Characters that are not (^) alphanumeric/emoji (\w) or whitespace (\s)


################################################################
#################### TWITTER SYNTAX FILTERS ####################
################################################################

def filter_hashtags(str):
    return re.sub(r'#\S+(\s|$)', '', str) # Hashtag followed by 1 or more (+) non-space characters (\S), then space (\s) or (|) end of line ($)

def filter_mentions(str):
    return re.sub(r'@\S+(?=(\s|$))', '', str) # @, followed by 1 or more (+) non-space characters (\S), then non-matched (?=) space (\s) or (|) end of line ($)

def filter_tweet_syntax(str):
    str = filter_mentions(str)
    str = filter_hashtags(str)
    if str[0:3] == 'RT ': # retweets begin with 'RT @...'
        str = str[3:]
    return str


####################################################################
######################## GRAMMATICAL FILTERS #######################
####################################################################

def remove_stopwords(tokens, stop_words):
    return [token for token in tokens if token not in stop_words]

# Changes words from plural form to singular form
lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Removes stems from words, e.g. 'flying' -> 'fly'
destemmer = PorterStemmer()
def destem_tokens(tokens):
    return [destemmer.stem(token) for token in tokens]


####################################################################
####################### OTHER STRING FILTERS #######################
####################################################################

def filter_links(str):
    return re.sub(r'(http|www)\S+(\s|$)', '', str) # http or (|) www, followed by 1 or more (+) non-space chars (\S), then space (\s) or (|) end of line ($)

# Removes numbers and any symbols/punctuation surrounding them
def filter_numbers_symbols(str):
    return re.sub(r'([^\w\s]+)?\d+([^\w\s]+)?', '', str) # 1 or more (+) numbers (\d) maybe (?) surrounded by characters that are not (^) alphanumeric/emoji (\w) or whitespace (\s)

def filter_short_words(str, min_len=3):
    return ' '.join([word for word in str.split() if len(word) >= min_len])


# Cleans tweet string as is done in:
# "A Complete VADER-Based Sentiment Analysis of Bitcoin (BTC) Tweets during the Era of COVID-19"
def clean(str):
    str = filter_ellipse(str)
    str = filter_numbers_symbols(str)
    str = filter_links(str)
    str = filter_tweet_syntax(str)
    str = filter_extra_whitespace(str)
    if CLEAN_PUNC:
        str = filter_punctuation(str)
    if CLEAN_EMOJI:
        str = filter_nonascii(str)
    return str


# Tokenization as is done in:
# "A Complete VADER-Based Sentiment Analysis of Bitcoin (BTC) Tweets during the Era of COVID-19"
# Given a string, splits it by whitespace into groups of: alphanumeric/emoji, punctuation, or emoticons
def tokenize(str):
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


def get_processed_tweets(csv_loc, do_clean=True, nltk_split=True, do_destem=True, do_lemmatize=True, remove_sw=True, stopwords_loc=STOP_WORDS_LOC, max_num=-1):
    num_processed = 0
    with open(csv_loc, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        next(reader) # discard first csv row
        for line in reader:
            try:
                text = str(line['text'])
                if do_clean:
                    text = clean(text)
                tokens = word_tokenize(text) if nltk_split else tokenize(text)
                if do_destem:
                    tokens = destem_tokens(tokens)
                if do_lemmatize:
                    tokens = lemmatize_tokens(tokens)
                if remove_sw:
                    tokens = remove_stopwords(tokens, read_stop_words_file())

                # Yield, not a return outside of For loop, so that entire file isnt read into memory
                num_processed += 1
                if (num_processed == max_num):
                    return tokens
                yield tokens
            except:
                print('ERROR PROCESSING LINE:', line)



if __name__ == '__main__':
    get_processed_tweets('Bitcoin_tweets.csv')