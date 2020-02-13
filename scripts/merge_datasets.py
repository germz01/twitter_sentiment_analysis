import os
import pandas as pd

# MERGE THE DATASETS IN ONE TRAINING SET ######################################

tasks = ['A', 'B']

for task in tasks:
    datasets = list()

    for file in os.listdir('../data/{}/train'.format(task)):
        if file.split('.')[-1] == 'txt':
            ds = pd.read_csv('../data/{}/train/{}'.format(task, file),
                             index_col=False, sep='\t',
                             names=['id', 'label', 'tweet'] if task == 'A' else
                             ['id', 'topic', 'label', 'tweet'])
            if task == 'B':
                datasets.append(ds[ds.label.isin(['positive', 'negative'])])
            else:
                datasets.append(ds)

    f_ds = pd.concat(datasets, ignore_index=True)

    f_ds = f_ds.sample(frac=1).reset_index(drop=True)

    f_ds.drop(f_ds[f_ds.tweet == 'Not Available'].index, inplace=True)
    f_ds.drop_duplicates(subset='tweet', inplace=True)
    f_ds.drop(labels='id', axis=1, inplace=True)

# SAVE THE FINAL TRAINING SET AS A CSV FILE ###################################

    f_ds.to_csv(path_or_buf='../data/{}/merged_train.csv'.format(task),
                index=False)
