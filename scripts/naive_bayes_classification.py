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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def remove_alpha(tweet):
    return " ".join([token for token in tokenizer.tokenize(tweet)
                    if token.isalpha()])


c_type = input('CLASSIFICATION TYPE[SO, PN, A]: ')

trainset = pd.read_csv('../data/train.csv')

if c_type == 'SO':
    trainset.label = trainset.label.\
      map({'positive': 1, 'negative': 1, 'neutral': 2})
elif c_type == 'PN':
    trainset = trainset[trainset.label != 'neutral']
    trainset.label = trainset.label.map({'positive': 1, 'negative': 2})
else:
    trainset.label = trainset.label.\
      map({'positive': 1, 'negative': 2, 'neutral': 3})

X_train, X_test, y_train, y_test = \
  train_test_split(trainset.tweet, trainset.label, test_size=0.2)


tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False,
                           reduce_len=True)

vectorizer = CountVectorizer(preprocessor=remove_alpha,
                             tokenizer=tokenizer.tokenize,
                             stop_words=stopwords.words('english'))
transformer = TfidfTransformer()
selectioner = SelectKBest(chi2)

learner = MultinomialNB()

pipe = Pipeline([('vect', vectorizer), ('trans', transformer),
                ('select', selectioner), ('bayes', learner)])

parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
              'trans__sublinear_tf': (False, True),
              'select__k': (1000, 2000, 5000),
              'bayes__alpha': tuple(np.linspace(0.1, 1., 3))}

classifier = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=3,
                          scoring='f1_macro', cv=3)
classifier.fit(X_train.values, y_train.values)

best_parameters = classifier.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# PREDICTION ##################################################################

predictions = classifier.predict(X_test.values)

pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).\
    transpose().to_csv('../results/classification_report_bayes_{}.csv'.
                       format(c_type))

