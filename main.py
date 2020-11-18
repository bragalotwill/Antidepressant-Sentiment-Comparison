# Reference: https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/

from webbot import Browser 
import time
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
from os import path
sns.set(style='darkgrid', context='talk', palette='Dark2')


def process(comment):
	return [t.lower() for t in nltk.word_tokenize(comment) if t.lower() not in stopwords.words('english') and t.isalpha()]

antideps = dict(map(lambda x: (x, []), ['lexapro', 'celexa', 'prozac', 'luvox', 'paxil', 'zoloft', 'cymbalta', 'effexor', 'pristiq', 'fetzima', 'amoxapine', 'anafranil', 'norpramin', 'sinequan', 'tofranil', 'pamelor', 'vivactil', 'surmontil']))
rewrite = False

if path.exists('antideps.pk1') and not rewrite:
	read = open('antideps.pk1', 'rb')
	antideps = pickle.load(read)
	read.close()

else:
	web = Browser()
	num_elem = 15
	scores = []
	for antidep in antideps.keys():
		web.go_to('google.com') 
		web.type('reddit ' + antidep)
		web.press(web.Key.ENTER)
		links = min(num_elem, len(web.find_elements('reddit.com', tag ='h3')))
		for link in range(1, links + 1):
			web.click('reddit.com', tag='h3', number=link)
			el = web.find_elements(tag='p')
			antideps[antidep] += list(map(lambda x: x.text, el))
			web.go_to('google.com') 
			web.type('reddit ' + antidep)
			web.press(web.Key.ENTER)
	write = open('antideps.pk1', 'wb')
	pickle.dump(antideps, write)
	write.close()

sia = SIA()

scores = []
for antidep in antideps.keys():
	res = []
	for comment in antideps[antidep]:
		pol = sia.polarity_scores(comment)
		pol['comment'] = comment
		res.append(pol)
	data = pd.DataFrame.from_records(res)
	data['label'] = 0
	data.loc[data['compound'] > .5, 'label'] = 1
	data.loc[data['compound'] < -.5, 'label'] = -1
	print(antidep + ': \n')
	pos = process(' '.join(list(data[data.label == 1].comment)))
	pos_freq = nltk.FreqDist(pos)
	print('Positive:')
	print(pos_freq.most_common(10))
	neg = process(' '.join(list(data[data.label == -1].comment)))
	neg_freq = nltk.FreqDist(neg)
	print('\nNegative:')
	print(neg_freq.most_common(10))
	scores.append((data.label.value_counts(normalize=True) * 100)[1])
	print('\nPercent Positive:', scores[-1], '\n')

fig, ax = plt.subplots(figsize=(20, 10), dpi=40)
sns.barplot(x=np.array(list(antideps.keys())), y=np.array(scores), ax=ax)
ax.set_xticklabels(np.array(list(antideps.keys())))
ax.set_ylabel("Percent Positive")
plt.show()