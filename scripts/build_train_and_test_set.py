import os
import pandas as pd

# BUILD THE FINAL DATASET THAT WILL BE USED FOR TRAINING AND TESTING ##########

dirs = [x for x in os.walk('../data/')][0][1]

datasets_tr, datasets_ts = [], []

for year in dirs:
    name = 'twitter_{}_train_A.txt'.format(year)
    ds = pd.read_csv('../data/' + year + '/' + name, index_col=False, sep='\t',
                     names=['id', 'label', 'tweet'])
    datasets_tr.append(ds)

    try:
        ds = \
            pd.read_csv('../data/' + year + '/' + name.replace('train', 'dev'),
                        index_col=False, sep='\t',
                        names=['id', 'label', 'tweet'])
        datasets_tr.append(ds)
    except Exception as e:
        pass

    try:
        ds = pd.read_csv('../data/' + year + '/' +
                         name.replace('train', 'test'), sep='\t',
                         index_col=False, names=['id', 'label', 'tweet'])
        datasets_ts.append(ds)
    except Exception as e:
        pass

for i, datasets in enumerate([datasets_tr, datasets_ts]):
    name = 'train.csv' if i == 0 else 'test.csv'

    f_ds = pd.concat(datasets, ignore_index=True)
    f_ds.drop(f_ds[f_ds.tweet == 'Not Available'].index, inplace=True)
    f_ds.drop_duplicates(subset='tweet', inplace=True)
    f_ds.drop(labels='id', axis=1, inplace=True)

# SAVE THE FINAL TRAINING AND TESTING DATASETs AS A CSV FILE ##################

    f_ds.to_csv(path_or_buf='../data/' + name, index=False)
