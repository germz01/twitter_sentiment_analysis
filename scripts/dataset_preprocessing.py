import pandas as pd

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm

tag_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}


def get_wn_pos(token):
    return tag_dict.get(token[1][0].upper(), wordnet.NOUN)


# IMPORTING THE TRAINING SET AND THE TEST SET #################################

trainset = pd.read_csv('../data/train.csv')

# TOKENIZATION ################################################################

tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False,
                           reduce_len=True)

trainset.tweet = [tokenizer.tokenize(tweet) for tweet in
                  tqdm(trainset.tweet, desc='TOKENIZATION')]

# POS TAGGING #################################################################

trainset.tweet = [pos_tag(tweet) for tweet in
                  tqdm(trainset.tweet, desc='POS TAGGING')]

# STOP WORD CLEANING AND REDUCTION TO ONLY ALPHANUMERIC TOKENS ################

stopwords = stopwords.words('english')

trainset.tweet = [[token for token in tweet if token[0].isalpha() and
                  token[0] not in stopwords] for tweet in
                  tqdm(trainset.tweet, desc='CLEANING')]

# LEMMATIZATION ###############################################################

lemmatizer = WordNetLemmatizer()

trainset.tweet = [[lemmatizer.lemmatize(token[0], get_wn_pos(token)) for
                  token in tweet] for tweet in
                  tqdm(trainset.tweet, desc='LEMMATIZATION')]

# REMOVING WORDS THAT ARE PRESENT IN ALL THE LABELS ###########################

tokens_set_per_label = dict()

for label in ['positive', 'negative', 'neutral']:
    total_tokens_list = list()

    for token_list in trainset[trainset.label == label].tweet:
        for token in token_list:
            total_tokens_list.append(token)

    tokens_set_per_label[label] = set(total_tokens_list)

intersection = tokens_set_per_label['positive'].\
    intersection(tokens_set_per_label['negative']).\
    intersection(tokens_set_per_label['neutral'])

trainset.tweet.apply(lambda x: [token for token in x if token not in
                     intersection])

# SAVING THE PREPROCESSED DATASETS ############################################

trainset.to_csv('../data/preprocessed_train.csv', index=False)
