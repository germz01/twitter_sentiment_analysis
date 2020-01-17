import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

trainset = pd.read_csv('../data/train.csv')
testset = pd.read_csv('../data/test.csv')

tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False,
                           reduce_len=True)
stop_words = stopwords.words('english')

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
selectioner = SelectKBest(chi2, k=5000)

# PREPROCESSING ###############################################################

trainset.tweet = [tokenizer.tokenize(tweet) for tweet in trainset.tweet]
trainset.tweet = [[token for token in tweet if token.isalpha()]
                  for tweet in trainset.tweet]
trainset.tweet = [[token for token in tweet if token not in stop_words]
                  for tweet in trainset.tweet]
trainset.tweet = [' '.join(tweet) for tweet in trainset.tweet]

testset.tweet = [tokenizer.tokenize(tweet) for tweet in testset.tweet]
testset.tweet = [[token for token in tweet if token.isalpha()]
                 for tweet in testset.tweet]
testset.tweet = [[token for token in tweet if token not in stop_words]
                 for tweet in testset.tweet]
testset.tweet = [' '.join(tweet) for tweet in testset.tweet]

X_train = vectorizer.fit_transform(trainset.tweet.values)
X_test = vectorizer.fit_transform(testset.tweet.values)
y_train = trainset.label.map({'positive': 1, 'negative': 2, 'neutral': 3})
y_test = testset.label.map({'positive': 1, 'negative': 2, 'neutral': 3})

X_train = selectioner.fit_transform(X_train, y_train)
X_test = selectioner.fit_transform(X_test, y_test)

X_train = transformer.fit_transform(X_train)
X_test = transformer.fit_transform(X_test)

# PREDICTION ##################################################################

learner = LinearSVC()
classifier = learner.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print(classification_report(y_test, predictions))
