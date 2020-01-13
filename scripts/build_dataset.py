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

datasets = []

for year in dirs:
    name = 'twitter_{}_train_A.txt'.format(year)
    ds = pd.read_csv(path_to_ds + year + '/' + name, sep='\t',
                     names=['id', 'label', 'tweet'])
    datasets.append(ds)

f_ts = pd.concat(datasets, ignore_index=True)
f_ts.drop(f_ts[f_ts.tweet == 'Not Available'].index, inplace=True)
f_ts.drop_duplicates(subset='tweet', inplace=True)
f_ts.drop(labels='id', axis=1, inplace=True)

# REMOVE TAGS, MENTIONS AND LINKS FROM THE TWEETS #############################

f_ts.tweet = f_ts.tweet.apply(clean_mentions_tags_and_links)

# SAVE THE FINAL TRAINING DATASET AS A CSV FILE ###############################

f_ts.to_csv(path_or_buf=path_to_ds + 'training_set.csv', index=False)
