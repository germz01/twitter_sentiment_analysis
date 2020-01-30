import numpy as np
import pandas as pd
import warnings

from gensim.corpora import Dictionary
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau, \
    CSVLogger
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# IMPORTING ###################################################################

trainset = pd.read_csv(
    '../data/preprocessed_train.csv',
    converters={'tweet': lambda x: x[1:-1].replace("'", "").split(', ')})

trainset, testset = train_test_split(trainset, test_size=0.2)

# VECTORIZE THE TWEETS COMPOSING THE CORPUS ###################################

mapping = input('WHICH MAPPING[SO/PN/A]: ')
assert mapping in ['SO', 'PN', 'A']

labels, labels_index, new_map = None, None, None

if mapping == 'SO':
    new_map = {'positive': 0, 'negative': 0, 'neutral': 1}
elif mapping == 'PN':
    new_map = {'positive': 0, 'negative': 1}
    trainset = trainset[trainset.label != 'neutral']
    testset = testset[testset.label != 'neutral']
else:
    new_map = {'positive': 0, 'negative': 1, 'neutral': 2}

trainset.label = trainset.label.map(new_map)
testset.label = testset.label.map(new_map)
train_labels, test_labels = trainset.label.values, testset.label.values
labels_index = new_map

word_dict = Dictionary(list(np.concatenate((trainset.tweet.values,
                                            testset.tweet.values))))

tokenizer = Tokenizer(num_words=len(word_dict))
tokenizer.fit_on_texts(list(np.concatenate((trainset.tweet.values,
                                           testset.tweet.values))))

X_train = tokenizer.texts_to_sequences(trainset.tweet.values)
X_test = tokenizer.texts_to_sequences(testset.tweet.values)

max_token_list_len = max([len(token_list) for token_list in
                         np.concatenate((trainset.tweet.values,
                                         testset.tweet.values))])

X_train = pad_sequences(X_train, maxlen=max_token_list_len)
X_test = pad_sequences(X_test, maxlen=max_token_list_len)

y_train = to_categorical(np.asarray(train_labels))
y_test = to_categorical(np.asarray(test_labels))

num_words = len(word_dict) + 1

# TRAINING THE MODEL ##########################################################

model = Sequential()

model.add(Embedding(num_words, 100, embeddings_initializer='glorot_uniform',
                    input_length=max_token_list_len))
# model.add(SpatialDropout1D(0.4))
model.add(LSTM(8        , dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2 if mapping != 'A' else 3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

log_name = '../results/classification_report_lstm_training_{}.csv'.\
    format(mapping)

history = model.fit(X_train, y_train, batch_size=128, epochs=5,
                    validation_split=0.3,
                    callbacks=[EarlyStopping(), ReduceLROnPlateau(),
                               CSVLogger(filename=log_name)],
                    shuffle=True)

# EVALUATING THE MODEL ########################################################

print()

log_name = '../results/classification_report_lstm_testing_{}.csv'.\
    format(mapping)

score = model.evaluate(X_test, y_test, verbose=1)

print('\nScore: {}, Accuracy: {}\n'.format(score[0], score[1]))

pd.DataFrame(np.array([score]), columns=['Score', 'Accuracy']).\
    to_csv('../results/classification_report_lstm_testing_{}.csv'.
           format(mapping), index=False)
