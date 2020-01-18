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

tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False,
                           reduce_len=True)

vectorizer = CountVectorizer(preprocessor=remove_alpha,
                             tokenizer=tokenizer.tokenize,
                             stop_words=stopwords.words('english'))
transformer = TfidfTransformer()
selectioner = SelectKBest(chi2, k=5000)

learner = LinearSVC()

pipe = Pipeline([('vect', vectorizer), ('trans', transformer),
                ('select', selectioner), ('svm', learner)])

parameters = {'svm__dual': (True, False),
              'svm__max_iter': (1000, 2000, 5000)}

grid_search = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=1)
grid_search.fit(trainset.tweet.values, trainset.label.values)

best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# PREPROCESSING ###############################################################

# X_train = vectorizer.fit_transform(trainset.tweet.values)
# X_test = vectorizer.fit_transform(testset.tweet.values)
# y_train = trainset.label.map({'positive': 1, 'negative': 2, 'neutral': 3})
# y_test = testset.label.map({'positive': 1, 'negative': 2, 'neutral': 3})

# X_train = selectioner.fit_transform(X_train, y_train)
# X_test = selectioner.fit_transform(X_test, y_test)

# X_train = transformer.fit_transform(X_train)
# X_test = transformer.fit_transform(X_test)

# PREDICTION ##################################################################

# classifier = learner.fit(X_train, y_train)
# predictions = classifier.predict(X_test)

# print(classification_report(y_test, predictions))
