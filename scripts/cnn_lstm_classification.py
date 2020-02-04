import codecs
import numpy as np
import pandas as pd
import warnings

from gensim.corpora import Dictionary
from keras.initializers import Constant
from keras.callbacks.callbacks import EarlyStopping
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")

# DEFINING THE MODEL TO BE VALIDATED VIA SKLEARN ##############################


def model_cnn(learning_rate=0.01):
    model = Sequential()

    if embeddings_index is None:
        model.add(Embedding(num_words, 100,
                            embeddings_initializer='glorot_uniform',
                            input_length=max_token_list_len))
    else:
        model.add(Embedding(num_words, 100 if embedding_type == 'G' else 300,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_token_list_len, trainable=False))

    model.add(Conv1D(8, 3, strides=1, padding='valid',
                     data_format='channels_last', activation='relu',
                     use_bias=True, kernel_initializer='glorot_uniform'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(8, activation='tanh',
                    kernel_initializer='glorot_uniform'))
    model.add(Dense(2 if mapping != 'A' else 3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=learning_rate),
                  metrics=['acc'])

    return model


def model_lstm(learning_rate=0.01, dropout=0.2, recurrent_dropout=0.2):
    model = Sequential()

    if embeddings_index is None:
        model.add(Embedding(num_words, 100,
                            embeddings_initializer='glorot_uniform',
                            input_length=max_token_list_len))
    else:
        model.add(Embedding(num_words, 100 if embedding_type == 'G' else 300,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_token_list_len, trainable=False))
    model.add(LSTM(8, dropout=dropout,
                   recurrent_dropout=recurrent_dropout))
    model.add(Dense(2 if mapping != 'A' else 3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=learning_rate),
                  metrics=['acc'])

    return model


# IMPORTING ###################################################################

trainset = pd.read_csv(
    '../data/preprocessed_train.csv',
    converters={'tweet': lambda x: x[1:-1].replace("'", "").split(', ')})

trainset, testset = train_test_split(trainset, test_size=0.2)

# CHOOSING AND INDEXING WORD VECTORS ##########################################

embedding_type = input('WHICH WORD EMBEDDING[A/G/F]: ')
assert embedding_type in ['A', 'F', 'G']

embeddings_index, e_path = None, '../pretrained_word_embeddings/'

if embedding_type == 'G':
    embeddings_index, word_embedding = dict(), 'glove/glove.6B.100d.txt'

    with open(e_path + word_embedding) as f:
        for line in tqdm(f, desc='INDEXING WORD VECTORS', total=400000):
            word, coeff = line.split(maxsplit=1)
            coeff = np.fromstring(coeff, 'f', sep=' ')
            embeddings_index[word] = coeff
elif embedding_type == 'F':
    embeddings_index, word_embedding = dict(), 'fasttext/wiki-news-300d-1M.vec'

    with codecs.open(e_path + word_embedding) as f:
        for line in tqdm(f, desc='INDEXING WORD VECTORS', total=999995):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coeff = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coeff
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

# PREPARATION OF THE EMBEDDING MATRIX #########################################

num_words = len(word_dict) + 1
embedding_matrix = None if embeddings_index is None else \
    np.zeros((num_words, 100 if embedding_type == 'G' else 300))

if embeddings_index is not None:
    for i, word in tqdm(tokenizer.index_word.items(),
                        desc='PREPARING EMBEDDING MATRIX'):
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# TRAINING/VALIDATING THE MODEL ###########################################

m_type = input('WHICH MODEL[CNN/LSTM]: ')
assert m_type in ['CNN', 'LSTM']

validation = input('VALIDATION[Y/N]? ')
assert validation in ['Y', 'N']

model = None

if validation == 'Y':
    model = KerasClassifier(build_fn=model_cnn if m_type == 'CNN'
                            else model_lstm, verbose=1, epochs=3)

    parameters = {'learning_rate': uniform(loc=0.001, scale=0.009),
                  'batch_size': randint(8, 33)}
    if m_type == 'LSTM':
        parameters['dropout'] = uniform(loc=0.2, scale=0.3)
        parameters['recurrent_dropout'] = uniform(loc=0.2, scale=0.3)
        parameters['batch_size'] = [32]

    n_iter = int(input('NUMBER OF POINTS TO TEST: '))

    classifier = RandomizedSearchCV(model, parameters, n_jobs=1, verbose=3,
                                    cv=3, refit=False, n_iter=n_iter)
    classifier.fit(X_train, y_train)

    best_parameters = classifier.best_params_

    model = model_cnn(learning_rate=best_parameters['learning_rate']) \
        if m_type == 'CNN' \
        else model_lstm(learning_rate=best_parameters['learning_rate'],
                        dropout=best_parameters['dropout'],
                        recurrent_dropout=best_parameters['recurrent_dropout'])

    history = model.fit(X_train, y_train, epochs=5,
                        validation_split=0.3, shuffle=True,
                        batch_size=best_parameters['batch_size'],
                        callbacks=[EarlyStopping()])
else:
    learning_rate_dist = uniform(loc=0.001, scale=0.009)
    dropout_dist = uniform(loc=0.2, scale=0.3)

    model = model_cnn(learning_rate=learning_rate_dist.rvs(1)[0]) \
        if m_type == 'CNN' \
        else model_lstm(learning_rate=learning_rate_dist.rvs(1)[0],
                        dropout=dropout_dist.rvs(1)[0],
                        recurrent_dropout=dropout_dist.rvs(1)[0])

    history = model.fit(X_train, y_train, epochs=5,
                        validation_split=0.3, shuffle=True,
                        callbacks=[EarlyStopping()])

# PREDICTION ##################################################################

log_name = '../results/classification_report_{}_{}_{}.csv'.\
    format(m_type.lower(), mapping, embedding_type)

score = model.evaluate(X_test, y_test, verbose=1)

print('\nScore: {}, Accuracy: {}\n'.format(score[0], score[1]))

pd.DataFrame(np.array([score]), columns=['Score', 'Accuracy']).\
    to_csv(log_name, index=False)
