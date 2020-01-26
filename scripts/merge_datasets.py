import os
import pandas as pd

# MERGE THE DATASETS IN ONE TRAINING SET ######################################

dirs = [x for x in os.walk('../data/')][0][1]

datasets_tr = list()

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
        datasets_tr.append(ds)
    except Exception as e:
        pass

f_ds = pd.concat(datasets_tr, ignore_index=True)

f_ds = f_ds.sample(frac=1).reset_index(drop=True)

f_ds.drop(f_ds[f_ds.tweet == 'Not Available'].index, inplace=True)
f_ds.drop_duplicates(subset='tweet', inplace=True)
f_ds.drop(labels='id', axis=1, inplace=True)

# SAVE THE FINAL TRAINING SET AS A CSV FILE ###################################

f_ds.to_csv(path_or_buf='../data/train.csv', index=False)
