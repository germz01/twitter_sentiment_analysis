import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from scipy.stats import randint, uniform
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def remove_alpha(tweet):
    return " ".join([token for token in tokenizer.tokenize(tweet)
                    if token.isalpha()])


t_type = input('TASK TYPE[A/B]: ')
assert t_type in ['A', 'B']

c_type = None if t_type == 'B' else input('CLASSIFICATION TYPE[SO, PN, A]: ')
assert c_type in [None, 'SO', 'PN', 'A']

# IMPORTING ###################################################################

trainset = pd.read_csv('../data/A/merged_train.csv') if t_type == 'A' else \
  pd.read_csv('../data/B/merged_train.csv')

testset = None
if t_type == 'A':
    testset = pd.read_csv('../data/A/test/2016test-A-English.txt', sep='\t',
                          index_col=False, names=['id', 'label', 'tweet'])
else:
    testset = pd.read_csv('../data/B/test/2016test-BD-English.txt', sep='\t',
                          index_col=False, names=['id', 'topic', 'label',
                                                  'tweet'])

# PREPROCESSING ###############################################################

if t_type == 'A':
    if c_type == 'SO':
        trainset.label = trainset.label.\
          map({'positive': 1, 'negative': 1, 'neutral': 2})
        testset.label = testset.label.\
            map({'positive': 1, 'negative': 1, 'neutral': 2})
    elif c_type == 'PN':
        trainset = trainset[trainset.label != 'neutral']
        trainset.label = trainset.label.map({'positive': 1, 'negative': 2})
    else:
        trainset.label = trainset.label.\
          map({'positive': 1, 'negative': 2, 'neutral': 3})
        testset.label = testset.label.\
            map({'positive': 1, 'negative': 2, 'neutral': 3})
else:
    trainset.label = trainset.label.map({'positive': 1, 'negative': 2})
    testset.label = testset.label.map({'positive': 1, 'negative': 2})

# VALIDATION ##################################################################

l_type = input('LEARNER[SVM/NB]: ')
assert l_type in ['SVM', 'NB']

X_train, y_train = trainset.tweet, trainset.label

tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False,
                           reduce_len=True)

vectorizer = CountVectorizer(preprocessor=remove_alpha,
                             tokenizer=tokenizer.tokenize,
                             stop_words=stopwords.words('english'))
transformer = TfidfTransformer()
selectioner = SelectKBest(chi2)

learner = LinearSVC() if l_type == 'SVM' else MultinomialNB()

pipe = Pipeline([('vect', vectorizer), ('trans', transformer),
                ('select', selectioner), (l_type.lower(), learner)])

parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
              'trans__sublinear_tf': (False, True),
              'select__k': randint(1000, 5000)}

if l_type == 'SVM':
    parameters['svm__C'] = uniform()
else:
    parameters['nb__alpha'] = uniform()

classifier = RandomizedSearchCV(pipe, parameters, n_iter=20, n_jobs=-1,
                                verbose=3, scoring='f1_weighted', cv=3)
classifier.fit(X_train.values, y_train.values)

# PREDICTION ##################################################################

if t_type == 'A':
    X_test, y_test = testset.tweet, testset.label

    predictions = classifier.predict(X_test.values)
    report = classification_report(y_test.values, predictions,
                                   output_dict=True)

    score = None

    if c_type == 'A':
        score = (report['1']['f1-score'] + report['2']['f1-score']) / 2
    else:
        score = (report['1']['recall'] + report['2']['recall']) / 2

    pd.DataFrame([score], columns=['Score']).\
        to_csv(path_or_buf='../results/A/{}_{}_score.csv'.format(l_type,
                                                                 c_type),
               index=False)
else:
    output_dataframe = pd.DataFrame(columns=['topic', 'score'])

    for topic in testset.topic.unique():
        topic_testset = testset[testset.topic == topic]
        X_test, y_test = topic_testset.tweet, topic_testset.label

        predictions = classifier.predict(X_test.values)
        report = classification_report(y_test.values, predictions,
                                       output_dict=True)

        recall_pn = None

        if '1' in report.keys():
            if '2' in report.keys():
                recall_pn = (report['1']['recall'] + report['2']['recall']) / 2
            else:
                recall_pn = (report['1']['recall'] + report['1']['recall']) / 2
        else:
            recall_pn = (report['2']['recall'] + report['2']['recall']) / 2

        output_dataframe = output_dataframe.append({'topic': topic,
                                                   'score': recall_pn},
                                                   ignore_index=True)

    output_dataframe.append({'topic': 'mean',
                            'score': np.mean(output_dataframe.score.values)},
                            ignore_index=True).\
        to_csv(path_or_buf='../results/B/{}_score.csv'.format(l_type),
               index=False)
