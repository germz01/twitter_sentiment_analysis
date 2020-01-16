import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag

"""
VISUALIZZAZIONI: VALORE DI P PER POST POSITIVI VS POST NEGATIVI

                 PAROLE PIU' USATE PER POST NEGATIVI, POSITIVI E NEUTRI

                 WORD CLOUD

"""


def clean_tweet_text(tweet):
    tweet = tokenizer.tokenize(tweet)
    tweet = [token for token in tweet if token.isalpha()]
    tweet = [token for token in tweet if token not in stop_words]

    return tweet


ds = pd.read_csv('../data/train.csv')

tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False,
                           reduce_len=True)
stop_words = stopwords.words('english')

# DISTRIBUTION OF THE WORD FREQUENCIES ########################################

c = Counter()

for tweet in ds.tweet:
    c.update(tweet.lower().split())

c = dict(Counter(c.values()))

plt.scatter(sorted(c), [c[k] for k in sorted(c)], marker='+', alpha=0.8)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Words')
plt.ylabel('Frequencies')
plt.title('Distribution of the word frequencies')
plt.savefig('../images/word_frequencies.png')
plt.close()

# TEXT CLEANING ###############################################################

ds.tweet = ds.tweet.apply(clean_tweet_text)

# TAG DISTRIBUTION FOR POSITIVE TWEETS VS NEGATIVE TWEETS #####################

samples = dict()
tags_counter = defaultdict(list)

for label in ['positive', 'negative']:
    samples[label] = ds[ds.label == label].sample(1000, random_state=42)
    samples[label]['tags'] = [pos_tag(t) for t in samples[label].tweet]

    for tags in samples[label].tags:
        for tag in tags:
            tags_counter[label].append(tag[1])

tags_counter = {'positive': Counter(tags_counter['positive']),
                'negative': Counter(tags_counter['negative'])}
tags_polarity = dict()

for tag in set(list(tags_counter['positive'].keys()) +
               list(tags_counter['negative'].keys())):

    num = (tags_counter['positive'][tag] + 1) - \
        (tags_counter['negative'][tag] + 1)
    den = (tags_counter['positive'][tag] + 1) + \
        (tags_counter['negative'][tag] + 1)
    tags_polarity[num / den] = tag

plt.figure(figsize=(15, 10))
bars = plt.bar(range(len(list(tags_polarity.keys()))),
               sorted(list(tags_polarity.keys())), zorder=5)
plt.grid(axis='y', zorder=3)

sk = sorted(list(tags_polarity.keys()))

for i, rect in enumerate(bars):
    plt.text(rect.get_x() + rect.get_width() / 2.0,
             rect.get_height() + 0.01 if rect.get_height() >= 0 else
             rect.get_height() - 0.03, tags_polarity[sk[i]], ha='center',
             va='bottom')

plt.xticks([])
plt.xlabel('Tags')
plt.ylabel('Polarity')
plt.title('Tag distribution: positive tweets VS negative tweets')
plt.savefig('../images/tag_distribution_positive_vs_negative.png')
plt.close()

# TAG DISTRIBUTION FOR OBJECTIVE TWEETS VS SUBJECTIVE TWEETS ##################

samples = dict()
tags_counter = defaultdict(list)

for label in ['objective', 'subjective']:
    if label == 'objective':
        samples[label] = ds[ds.label == 'neutral'].sample(1000,
                                                          random_state=42)
    else:
        samples[label] = ds[ds.label != 'neutral'].sample(1000,
                                                          random_state=42)

    samples[label]['tags'] = [pos_tag(t) for t in samples[label].tweet]

    for tags in samples[label].tags:
        for tag in tags:
            tags_counter[label].append(tag[1])

tags_counter = {'objective': Counter(tags_counter['objective']),
                'subjective': Counter(tags_counter['subjective'])}
tags_polarity = dict()

for tag in set(list(tags_counter['objective'].keys()) +
               list(tags_counter['subjective'].keys())):

    num = (tags_counter['objective'][tag] + 1) - \
        (tags_counter['subjective'][tag] + 1)
    den = (tags_counter['objective'][tag] + 1) + \
        (tags_counter['subjective'][tag] + 1)
    tags_polarity[num / den] = tag

plt.figure(figsize=(15, 10))
bars = plt.bar(range(len(list(tags_polarity.keys()))),
               sorted(list(tags_polarity.keys())), zorder=5)
plt.grid(axis='y', zorder=3)

sk = sorted(list(tags_polarity.keys()))

for i, rect in enumerate(bars):
    plt.text(rect.get_x() + rect.get_width() / 2.0,
             rect.get_height() + 0.01 if rect.get_height() >= 0 else
             rect.get_height() - 0.03, tags_polarity[sk[i]], ha='center',
             va='bottom')

plt.xticks([])
plt.xlabel('Tags')
plt.ylabel('Polarity')
plt.title('Tag distribution: objective tweets VS subjective tweets')
plt.savefig('../images/tag_distribution_objective_vs_subjective.png')
plt.close()
