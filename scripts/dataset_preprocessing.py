import pandas as pd

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

tag_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}


def get_wn_pos(token):
    return tag_dict.get(token[1][0].upper(), wordnet.NOUN)


# IMPORTING THE TRAINING SET AND THE TEST SET #################################

print('IMPORTING')

trainset = pd.read_csv('../data/train.csv')
testset = pd.read_csv('../data/test.csv')

# TOKENIZATION ################################################################

print('TOKENIZATION')

tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False,
                           reduce_len=True)

trainset.tweet = [tokenizer.tokenize(tweet) for tweet in trainset.tweet]
testset.tweet = [tokenizer.tokenize(tweet) for tweet in testset.tweet]

# POS TAGGING #################################################################

print('POS TAGGING')

trainset.tweet = [pos_tag(tweet) for tweet in trainset.tweet]
testset.tweet = [pos_tag(tweet) for tweet in testset.tweet]

# STOP WORD CLEANING AND REDUCTION TO ONLY ALPHANUMERIC TOKENS ################

print('CLEANING')

stopwords = stopwords.words('english')

trainset.tweet = [[token for token in tweet if token[0].isalpha() and
                  token[0] not in stopwords] for tweet in trainset.tweet]
testset.tweet = [[token for token in tweet if token[0].isalpha() and
                  token[0] not in stopwords] for tweet in testset.tweet]

# LEMMATIZATION ###############################################################

print('LEMMATIZATION')

lemmatizer = WordNetLemmatizer()

trainset.tweet = [[lemmatizer.lemmatize(token[0], get_wn_pos(token)) for
                  token in tweet] for tweet in trainset.tweet]
testset.tweet = [[lemmatizer.lemmatize(token[0], get_wn_pos(token)) for
                  token in tweet] for tweet in testset.tweet]

# SAVING THE PREPROCESSED DATASETS ############################################

print('SAVING')

trainset.to_csv('../data/preprocessed_train.csv')
testset.to_csv('../data/preprocessed_test.csv')
