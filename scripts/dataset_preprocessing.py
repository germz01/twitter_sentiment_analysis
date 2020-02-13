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

trainset_A = pd.read_csv('../data/A/merged_train.csv')
trainset_B = pd.read_csv('../data/B/merged_train.csv')

# TOKENIZATION ################################################################

tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False,
                           reduce_len=True)

trainset_A.tweet = [tokenizer.tokenize(tweet) for tweet in
                    tqdm(trainset_A.tweet, desc='TOKENIZATION A')]
trainset_B.tweet = [tokenizer.tokenize(tweet) for tweet in
                    tqdm(trainset_B.tweet, desc='TOKENIZATION B')]

# POS TAGGING #################################################################

trainset_A.tweet = [pos_tag(tweet) for tweet in
                    tqdm(trainset_A.tweet, desc='POS TAGGING A')]
trainset_B.tweet = [pos_tag(tweet) for tweet in
                    tqdm(trainset_B.tweet, desc='POS TAGGING B')]

# STOP WORD CLEANING AND REDUCTION TO ONLY ALPHANUMERIC TOKENS ################

stopwords = stopwords.words('english')

trainset_A.tweet = [[token for token in tweet if token[0].isalpha() and
                    token[0] not in stopwords] for tweet in
                    tqdm(trainset_A.tweet, desc='CLEANING A')]
trainset_B.tweet = [[token for token in tweet if token[0].isalpha() and
                    token[0] not in stopwords] for tweet in
                    tqdm(trainset_B.tweet, desc='CLEANING B')]

# LEMMATIZATION ###############################################################

lemmatizer = WordNetLemmatizer()

trainset_A.tweet = [[lemmatizer.lemmatize(token[0], get_wn_pos(token)) for
                    token in tweet] for tweet in
                    tqdm(trainset_A.tweet, desc='LEMMATIZATION A')]
trainset_B.tweet = [[lemmatizer.lemmatize(token[0], get_wn_pos(token)) for
                    token in tweet] for tweet in
                    tqdm(trainset_B.tweet, desc='LEMMATIZATION B')]

# REMOVING WORDS THAT ARE PRESENT IN ALL THE LABELS ###########################

tokens_set_per_label = dict()

for label in ['positive', 'negative', 'neutral']:
    total_tokens_list = list()

    for token_list in trainset_A[trainset_A.label == label].tweet:
        for token in token_list:
            total_tokens_list.append(token)

    tokens_set_per_label[label] = set(total_tokens_list)

intersection = tokens_set_per_label['positive'].\
    intersection(tokens_set_per_label['negative']).\
    intersection(tokens_set_per_label['neutral'])

trainset_A.tweet.apply(lambda x: [token for token in x if token not in
                       intersection])

# SAVING THE PREPROCESSED DATASETS ############################################

trainset_A.to_csv('../data/A/preprocessed_train.csv', index=False)
trainset_B.to_csv('../data/B/preprocessed_train.csv', index=False)
