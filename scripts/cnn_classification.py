import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ipdb

from gensim.corpora import Dictionary
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau, \
    CSVLogger
from keras.initializers import Constant
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding
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
testset = pd.read_csv(
    '../data/preprocessed_test.csv',
    converters={'tweet': lambda x: x[1:-1].replace("'", "").split(', ')})

# CHOOSING AND INDEXING WORD VECTORS ##########################################

embedding_type = input('WHICH WORD EMBEDDING[A/G/F]: ')
assert embedding_type in ['A', 'F', 'G']

embeddings_index = None

if embedding_type == 'G':
    embeddings_index, word_embedding = dict(), 'glove/glove.6B.100d.txt'

    with open('../pretrained_word_embeddings/{}'.format(word_embedding)) as f:
        for line in tqdm(f, desc='INDEXING WORD VECTORS', total=400000):
            word, coeff = line.split(maxsplit=1)
            coeff = np.fromstring(coeff, 'f', sep=' ')
            embeddings_index[word] = coeff
elif embedding_type == 'F':
    # TO DO
    embeddings_index = dict()
else:
    pass

# VECTORIZE THE TWEETS COMPOSING THE CORPUS ###################################

mapping = input('WHICH MAPPING[SO/PN/A]: ')
assert mapping in ['SO', 'PN', 'A']

labels, labels_index, new_map = None, None, None

if mapping == 'SO':
    new_map = {'positive': 0, 'negative': 0, 'neutral': 1}
elif mapping == 'PN':
    new_map = {'positive': 0, 'negative': 1}
    trainset = trainset[trainset.label != 'neutral']
else:
    new_map = {'positive': 0, 'negative': 1, 'neutral': 2}

trainset.label = trainset.label.map(new_map)
testset.label = testset.label.map(new_map)
train_labels, test_labels = trainset.label.values, testset.label.values
labels_index = new_map

word_dict = Dictionary(list(trainset.tweet.values))

tokenizer = Tokenizer(num_words=len(word_dict))
tokenizer.fit_on_texts(trainset.tweet.values)

X_train = tokenizer.texts_to_sequences(trainset.tweet.values)
X_test = tokenizer.texts_to_sequences(testset.tweet.values)
X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)

y_train = to_categorical(np.asarray(train_labels))
y_test = to_categorical(np.asarray(test_labels))

# PREPARATION OF THE EMBEDDING MATRIX #########################################

num_words = len(word_dict) + 1
embedding_matrix = None if embeddings_index is None else \
    np.zeros((num_words, 100))

if embeddings_index is not None:
    for i, word in tqdm(tokenizer.index_word.items(),
                        desc='PREPARING EMBEDDING MATRIX'):
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# TRAINING THE MODEL ##########################################################

model = Sequential()

if embeddings_index is None:
    model.add(Embedding(num_words, 100,
                        embeddings_initializer='glorot_uniform',
                        input_length=100))
else:
    model.add(Embedding(num_words, 100,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=100, trainable=False))

model.add(Conv1D(128, 3, strides=1, padding='valid',
                 data_format='channels_last', activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform'))
model.add(GlobalMaxPooling1D())
model.add(Dense(16, activation='tanh', kernel_initializer='glorot_uniform'))
model.add(Dense(2 if mapping != 'A' else 3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.01),
              metrics=['acc'])

log_name = '../results/classification_report_cnn_training_{}_{}.csv'.\
    format(mapping, embedding_type)

history = model.fit(X_train, y_train, batch_size=128, epochs=5,
                    validation_split=0.3,
                    callbacks=[EarlyStopping(), ReduceLROnPlateau(),
                               CSVLogger(filename=log_name)],
                    shuffle=True)

# EVALUATING THE MODEL ########################################################

log_name = '../results/classification_report_cnn_testing_{}_{}.csv'.\
    format(mapping, embedding_type)

score = model.evaluate(X_test, y_test, verbose=1)

print('\nScore: {}, Accuracy: {}\n'.format(score[0], score[1]))

# PLOTTING ####################################################################

plt.plot(range(len(history.history['loss'])), history.history['loss'],
         label='TRAINING LOSS')
plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'],
         label='VALIDATION LOSS')
plt.xticks(range(len(history.history['loss'])),
           range(1, len(history.history['loss']) + 1))
plt.legend()
plt.grid()
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('../images/cnn_training_loss.png')
plt.close()

plt.plot(range(len(history.history['acc'])), history.history['acc'],
         label='TRAINING ACCURACY')
plt.plot(range(len(history.history['val_acc'])), history.history['val_acc'],
         label='VALIDATION ACCURACY')
plt.xticks(range(len(history.history['acc'])),
           range(1, len(history.history['acc']) + 1))
plt.legend()
plt.grid()
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('../images/cnn_training_accuracy.png')
plt.close()
