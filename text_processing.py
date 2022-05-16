from pickle import FALSE
import re, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Text processing parameters
DO_CLEAN = True
NLTK_SPLIT = True
DO_DESTEM = False
DO_LEMMATIZE = True
REMOVE_SW = True
STOP_WORDS_LOC = 'stop_words.txt'


def read_stop_words_file():
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



def remove_special_char(text):
    """remove special characters"""
    return re.sub(r"[^A-Za-z0-9\s]+", " ", text)

def remove_whitespace(text):
    return re.sub('r[...\n\d{2}$][\n+\d{2}$]',"", text)

def remove_parenthesis(text):
    """Remove parenthesis and text therein"""
    return re.sub(r'\([^)]*\)', '', text)


def remove_numbers(text):
    """Remove Numbers"""
    return re.sub(r"\b[0-9]+\b\s*","",text)

def remove_char(text):
    """Remove special characters"""
    return re.sub(r'[#,@,&,—,:,%, ©, ...]'," ",text)

def replace_slash(text):
    """Replaces slash with whitespace"""
    return re.sub(r"/", " ", text)

def remove_acronyms(text):
    """Remove all acronyms"""
    return re.sub(r'\b[A-Z]{2,}\b',"",text)

def lower_case(text):
    """Returns lowercase"""
    return str(text).lower()


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

def emoji_remover(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def strip_links(text):
    """Removes urls from text, then concats them back together"""
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    """Removes hashtag, @-symbol and text attached to it"""
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


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
    return str


# Curry function is a function nesting other functions.
# Because Pandas' ".apply()" method takes in a single function as its arg, 
# currying function allows for the application of multiple functions operating on each row in the df.
# Additionally, this approach is immutable, never augmenting the original data. 

def currying(special_char, acronyms, whitespace, numeric, number, parenthesis, char, slash, lower, emoji, hashtag, striplink, stripentity, shortwords):
    def text_cleaner(x):
        return whitespace(
                    shortwords(
                        char(
                            special_char(
                                number(
                                    numeric(
                                        parenthesis(
                                                slash(
                                                        acronyms(
                                                            emoji(
                                                                    hashtag(
                                                                            stripentity(
                                                                                striplink(
                                                                                    lower(x))))))))))))))
    
    return text_cleaner


# The variable below refers to the currying function and all its initalized args.
# When curry_text_cleaner is supplied as an arg for data["tokens"].apply(), 
# pandas will supply each row of the "tokens" column as an argument to the inner function of the currying() function,
# which is referred to as text_cleaner(x). The argument "x" inside of the inner function text_cleaner will represent
# each row in the "tokens column." Each of the 14 functions will be applied to the string the row.
curry_text_cleaner = currying(remove_special_char,
                     remove_acronyms,
                     remove_whitespace,
                     filter_numbers,
                     remove_numbers,
                     remove_parenthesis,
                     remove_char,
                     replace_slash,
                     lower_case,
                     emoji_remover,
                     filter_hashtags,
                     strip_links,
                     strip_all_entities,
                     filter_short_words)




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


def process_tweet(text, do_clean=DO_CLEAN, nltk_split=NLTK_SPLIT, do_destem=DO_DESTEM, do_lemmatize=DO_LEMMATIZE, remove_sw=REMOVE_SW):
    text = str(text).lower()
    if do_clean:
        text = clean(text)
    tokens = word_tokenize(text) if nltk_split else tokenize(text)
    if do_destem:
        tokens = destem_tokens(tokens)
    if do_lemmatize:
        tokens = lemmatize_tokens(tokens)
    if remove_sw:
        tokens = remove_stopwords(tokens, read_stop_words_file())
    return tokens
