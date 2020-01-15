import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter
from collections import defaultdict
from nltk import word_tokenize
from nltk import pos_tag

"""
VISUALIZZAZIONI: DISTRIBUZIONE DELLA FREQUENZA DELLE PAROLE PER VEDERE SE
                 VIENE SEGUITA LA LEGGE DI ZIPF

                 VALORE DI P PER POST SOGGETTIVI VS POST OGGETTIVI

                 VALORE DI P PER POST POSITIVI VS POST NEGATIVI

                 PAROLE PIU' USATE PER POST NEGATIVI, POSITIVI E NEUTRI

                 WORD CLOUD

"""

ds = pd.read_csv('../data/train.csv')

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

# TAG DISTRIBUTION FOR POSITIVE TWEETS VS NEGATIVE TWEETS #####################

positive_sample = ds[ds.label == 'positive'].sample(500, random_state=42)
negative_sample = ds[ds.label == 'negative'].sample(500, random_state=42)
samples = {'positive': positive_sample, 'negative': negative_sample}

tags_counter = defaultdict(list)

for label in ['positive', 'negative']:
    samples[label]['tags'] = [pos_tag(word_tokenize(t))
                              for t in samples[label].tweet]

    for tags in samples[label].tags:
        for tag in tags:
            tags_counter[label].append(tag[1])

tags_counter = {'positive': Counter(tags_counter['positive']),
                'negative': Counter(tags_counter['negative'])}
tags_polarity = dict()

for tag in set(list(tags_counter['positive'].keys()) +
               list(tags_counter['negative'].keys())):
    num = tags_counter['positive'][tag] - tags_counter['negative'][tag]
    den = tags_counter['positive'][tag] + tags_counter['negative'][tag]
    tags_polarity[num / den] = tag

bars = plt.bar([2 * i for i in range(len(list(tags_polarity.keys())))],
               sorted(list(tags_polarity.keys())), zorder=5)
plt.grid(axis='y', zorder=3)

sk = sorted(list(tags_polarity.keys()))

for i, rect in enumerate(bars):
    plt.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height(),
             tags_polarity[sk[i]], ha='center', va='bottom')

plt.xticks([])
plt.xlabel('Tags')
plt.ylabel('Polarity')
plt.title('Tag distribution: positive tweets VS negative tweets')
plt.savefig('../images/tag_distribution_positive_vs_negative.png')
plt.close()
