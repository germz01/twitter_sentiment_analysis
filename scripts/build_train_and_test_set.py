import os
import pandas as pd


def clean_mentions_tags_and_links(tweet):
    tokens = []

    for token in tweet.split():
        if (token[0] not in ['@', '#']) and ('http' not in token):
            tokens.append(token)

    return ' '.join(tokens)


# BUILD THE FINAL DATASET THAT WILL BE USED FOR TRAINING ######################

path_to_ds = '../data/DOWNLOAD/Subtask_A/'
dirs = [x for x in os.walk(path_to_ds)][0][1]

datasets_tr, datasets_ts = [], []

for year in dirs:
    name = 'twitter_{}_train_A.txt'.format(year)
    ds = pd.read_csv(path_to_ds + year + '/' + name, sep='\t',
                     names=['id', 'label', 'tweet'])
    datasets_tr.append(ds)

    ds = pd.read_csv(path_to_ds + year + '/' + name.replace('train', 'test'),
                     sep='\t', names=['id', 'label', 'tweet'])
    datasets_ts.append(ds)

for i, datasets in enumerate([datasets_tr, datasets_ts]):
    name = 'train.csv' if i == 0 else 'test.csv'

    f_ds = pd.concat(datasets, ignore_index=True)
    f_ds.drop(f_ds[f_ds.tweet == 'Not Available'].index, inplace=True)
    f_ds.drop_duplicates(subset='tweet', inplace=True)
    f_ds.drop(labels='id', axis=1, inplace=True)

# REMOVE TAGS, MENTIONS AND LINKS FROM THE TWEETS #############################

    f_ds.tweet = f_ds.tweet.apply(clean_mentions_tags_and_links)

# SAVE THE FINAL TRAINING DATASET AS A CSV FILE ###############################

    f_ds.to_csv(path_or_buf=path_to_ds + name, index=False)
