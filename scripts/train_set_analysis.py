import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter

"""
VISUALIZZAZIONI: DISTRIBUZIONE DELLA FREQUENZA DELLE PAROLE PER VEDERE SE
                 VIENE SEGUITA LA LEGGE DI ZIPF

                 VALORE DI P PER POST SOGGETTIVI VS POST OGGETTIVI

                 VALORE DI P PER POST POSITIVI VS POST NEGATIVI

                 PAROLE PIU' USATE PER POST NEGATIVI, POSITIVI E NEUTRI

                 WORD CLOUD

"""

ds = pd.read_csv('../data/train.csv')

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
