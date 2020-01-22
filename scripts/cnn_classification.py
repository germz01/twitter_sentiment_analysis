import ipdb
import numpy as np
import pandas as pd

from keras.initializers import Constant
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm

# INDEXING WORD VECTORS #######################################################

word_embedding = input('WHICH WORD EMBEDDING[G/F]: ')
assert word_embedding in ['F', 'G']
word_embedding = 'glove/glove.6B.100d.txt' if word_embedding == 'G' \
    else 'fasttext'


embeddings_index = {}

with open('../pretrained_word_embeddings/{}'.format(word_embedding)) as f:
    for line in tqdm(f, desc='INDEXING WORD VECTORS', total=400000):
        word, coeff = line.split(maxsplit=1)
        coeff = np.fromstring(coeff, 'f', sep=' ')
        embeddings_index[word] = coeff

# PREPROCESSING ###############################################################

trainset = pd.read_csv('../data/train.csv')

tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False,
                           reduce_len=True)
stopwords = stopwords.words('english')

trainset.tweet = [tokenizer.tokenize(tweet) for tweet in trainset.tweet]
trainset.tweet = [[token for token in tweet if token.isalpha() and
                  token not in stopwords] for tweet in trainset.tweet]

# VECTORIZE THE TWEETS COMPOSING THE CORPUS ###################################

mapping = input('WHICH MAPPING[SO/PN/A]: ')
assert mapping in ['SO', 'PN', 'A']

labels, labels_index, mapping = None, None, None

if mapping == 'SO':
    mapping = {'positive': 0, 'negative': 0, 'neutral': 1}
    labels = trainset.label.map(mapping).values
elif mapping == 'PN':
    mapping = {'positive': 0, 'negative': 1}
    labels = trainset[trainset.label != 'neutral'].label.map(mapping).values
else:
    mapping = {'positive': 0, 'negative': 1, 'neutral': 2}
    labels = trainset.label.map(mapping).values

labels_index = mapping

tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainset.tweet.values)

sequences = tokenizer.texts_to_sequences(trainset.tweet.values)

word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=1000)
labels = to_categorical(np.asarray(labels))

# SPLITTING THE CORPUS INTO A TRAINING SET AND A VALIDATION SET ###############

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

X_train = data[:-int(0.3 * data.shape[0])]
y_train = labels[:-int(0.3 * data.shape[0])]
X_val = data[-int(0.3 * data.shape[0]):]
y_val = labels[-int(0.3 * data.shape[0]):]

# PREPARATION OF THE EMBEDDING MATRIX #########################################

# num_words = min(20000, len(word_index) + 1)
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, 100))

for word, i in tqdm(word_index.items(), desc='PREPARING EMBEDDING MATRIX'):
    # if i >= 20000:
    #     continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words, 100,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=1000, trainable=False)

# TRAINING THE MODEL ##########################################################

sequence_input = Input(shape=(1000,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)

preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['acc'])

model.fit(X_train, y_train, batch_size=128, epochs=10,
          validation_data=(X_val, y_val))
