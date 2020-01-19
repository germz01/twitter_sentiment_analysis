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


trainset = pd.read_csv('../data/train.csv')
testset = pd.read_csv('../data/test.csv')

X_train = trainset.tweet.values
y_train = trainset.label.map({'positive': 1, 'negative': 2, 'neutral': 3}).\
    values
X_test = testset.tweet.values
y_test = testset.label.map({'positive': 1, 'negative': 2, 'neutral': 3}).values

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

parameters = {'trans__sublinear_tf': (False, True),
              'select__k': (1000, 2000, 5000),
              'svm__C': tuple(np.linspace(0.1, 1., 3))}

classifier = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=4,
                          scoring='f1_macro', cv=3)
classifier.fit(X_train, y_train)

best_parameters = classifier.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# PREDICTION ##################################################################

predictions = classifier.predict(X_test)

print(classification_report(y_test, predictions))
