import numpy as np
import pandas as pd

import ipdb

from gensim.corpora import Dictionary
from keras.callbacks.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tqdm import tqdm

# IMPORTING ###################################################################

trainset = pd.read_csv(
    '../data/preprocessed_train.csv',
    converters={'tweet': lambda x: x[1:-1].replace("'", "").split(', ')})

# CHOOSING AND INDEXING WORD VECTORS ##########################################

word_embedding = input('WHICH WORD EMBEDDING[A/G/F]: ')
assert word_embedding in ['A', 'F', 'G']

embeddings_index = {}

if word_embedding == 'G':
    word_embedding = 'glove/glove.6B.100d.txt'

    with open('../pretrained_word_embeddings/{}'.format(word_embedding)) as f:
        for line in tqdm(f, desc='INDEXING WORD VECTORS', total=400000):
            word, coeff = line.split(maxsplit=1)
            coeff = np.fromstring(coeff, 'f', sep=' ')
            embeddings_index[word] = coeff
elif word_embedding == 'F':
    # TO DO
    pass
else:
    pass

# VECTORIZE THE TWEETS COMPOSING THE CORPUS ###################################

mapping = input('WHICH MAPPING[SO/PN/A]: ')
assert mapping in ['SO', 'PN', 'A']

labels, labels_index = None, None

if mapping == 'SO':
    mapping = {'positive': 0, 'negative': 0, 'neutral': 1}
elif mapping == 'PN':
    mapping = {'positive': 0, 'negative': 1}
    trainset = trainset[trainset.label != 'neutral']
else:
    mapping = {'positive': 0, 'negative': 1, 'neutral': 2}

trainset.label = trainset.label.map(mapping)
labels = trainset.label.values
labels_index = mapping

tokenizer = Tokenizer(num_words=len(Dictionary(list(trainset.tweet.values))))
tokenizer.fit_on_texts(trainset.tweet.values)

sequences = tokenizer.texts_to_sequences(trainset.tweet.values)

word_index = tokenizer.word_index

X_train = pad_sequences(sequences,
                        maxlen=max([len(tl) for tl in trainset.tweet]))
y_train = to_categorical(np.asarray(labels))

# PREPARATION OF THE EMBEDDING MATRIX #########################################

# num_words = min(20000, len(word_index) + 1)
num_words = len(Dictionary(list(trainset.tweet.values))) + 1
# embedding_matrix = np.zeros((num_words, 100))

# for word, i in tqdm(word_index.items(), desc='PREPARING EMBEDDING MATRIX'):
#     # if i >= 20000:
#     #     continue

#     embedding_vector = embeddings_index.get(word)

#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector

# TRAINING THE MODEL ##########################################################

model = Sequential()

model.add(Embedding(num_words, 100, embeddings_initializer='glorot_uniform',
                    input_length=max([len(tl) for tl in trainset.tweet])))
model.add(Conv1D(5, 3, strides=1, padding='valid', data_format='channels_last',
                 activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform'))
model.add(MaxPooling1D(pool_size=3, strides=None, padding='valid',
                       data_format='channels_last'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(2 if mapping != 'A' else 3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['acc'])

history = model.fit(X_train, y_train, batch_size=16, epochs=10,
                    validation_split=0.3,
                    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0,
                                             patience=0, mode='min')],
                    shuffle=True)

