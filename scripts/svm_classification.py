import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def remove_alpha(tweet):
    return " ".join([token for token in tokenizer.tokenize(tweet)
                    if token.isalpha()])


c_type = input('CLASSIFICATION TYPE[SO, PN, A]: ')

trainset = pd.read_csv('../data/train.csv')
testset = pd.read_csv('../data/test.csv')

X_train, X_test, y_train, y_test = None, None, None, None

if c_type == 'SO':
    X_train = trainset.tweet.values
    X_test = testset.tweet.values
    y_train = trainset.label.\
        map({'positive': 1, 'negative': 1, 'neutral': 2}).values
    y_test = testset.label.\
        map({'positive': 1, 'negative': 1, 'neutral': 2}).values
elif c_type == 'PN':
    X_train = trainset[trainset.label != 'neutral'].tweet.values
    X_test = testset[testset.label != 'neutral'].tweet.values
    y_train = trainset[trainset.label != 'neutral'].label.\
        map({'positive': 1, 'negative': 2}).values
    y_test = testset[testset.label != 'neutral'].label.\
        map({'positive': 1, 'negative': 2}).values
else:
    X_train = trainset.tweet.values
    X_test = testset.tweet.values
    y_train = trainset.label.\
        map({'positive': 1, 'negative': 2, 'neutral': 3}).values
    y_test = testset.label.\
        map({'positive': 1, 'negative': 2, 'neutral': 3}).values


tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False,
                           reduce_len=True)

vectorizer = CountVectorizer(preprocessor=remove_alpha,
                             tokenizer=tokenizer.tokenize,
                             stop_words=stopwords.words('english'))
transformer = TfidfTransformer()
selectioner = SelectKBest(chi2)

learner = LinearSVC()

pipe = Pipeline([('vect', vectorizer), ('trans', transformer),
                ('select', selectioner), ('svm', learner)])

parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
              'trans__sublinear_tf': (False, True),
              'select__k': (1000, 2000, 5000),
              'svm__C': tuple(np.linspace(0.1, 1., 3))}

classifier = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=3,
                          scoring='f1_macro', cv=3)
classifier.fit(X_train, y_train)

best_parameters = classifier.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# PREDICTION ##################################################################

predictions = classifier.predict(X_test)

pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).\
    transpose().to_csv('./classification_report_svm_{}.csv'.format(c_type))
