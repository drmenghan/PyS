__author__ = 'mhan'

import nltk

# nltk.download()


sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
tokens
tagged = nltk.pos_tag(tokens)
tagged[0:6]

from nltk.book import *
text1.similar("monstrous")

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties","America"])
import matplotlib
matplotlib.use('Agg')

text3.plot()
len(text3)
sorted(set(text3))

len(set(text3))
len(text3) / len(set(text3))

text4[173]

saying = ['After', 'all', 'is', 'said', 'and', 'done','more', 'is', 'said', 'than', 'done']
tokens = set(saying)
tokens = sorted(tokens)
tokens[3:1]

fdist1 = FreqDist(text1)
vocabulary1 = fdist1.keys()

fdist1.plot(50, cumulative = True)

text4.collocations()

[len(w) for w in text1]

babelize_shell()

sent = ['she', 'sells', 'sea', 'shells', 'by','the', 'sea', 'shore']
for w in sent:
    if w[:2]=='sh':
        print(w)

import nltk
nltk.corpus.gutenberg.fileids()
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)
type(emma)

macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')

macbeth_sentences = gutenberg.sents('colon_delimited_stock_prices.txt')


from nltk.corpus import webtext
for fileid in webtext.fileids():
    print(fileid,webtext.raw(fileid)[:25])

from nltk.corpus import brown
brown.categories()


from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch']
cfd = nltk.ConditionalFreqDist((lang, len(word))
                               for lang in languages
                               for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative= True)

from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist((genre, word)
                               for genre in brown.categories()
                               for word in brown.words(categories = genre))
print(cfd)
# cfd.plot(cumulative= True)
cfd['news']


def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)

unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))


import nltk
def content_fraction(text):
    """

    :param text:
    :return: words not in stop-words
    """
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)

content_fraction(nltk.corpus.reuters.words())

len(stopwords)


names = nltk.corpus.names
names.fileids()
male_names = names.words('male.txt')
female_names = names.words('female.txt')
[w for w in male_names if w in female_names]

cfd = nltk.ConditionalFreqDist((fileid,name[-1])
                               for fileid in names.fileids()
                               for name in names.words(fileid))
cfd.plot()

from nltk.corpus import wordnet as wn
wn.synsets('motorcar')
wn.synset('car.n.01').lemma_names()
wn.synset('car.n.01').definition()
wn.synset('car.n.01').examples()
wn.lemmas('car')

from urllib import request
import urllib
url = "http://www.gutenberg.org/cache/epub/2554/pg2554.txt"
raw = request.urlopen(url)

from nltk.collocations import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw = open("pg2554.txt","rU").readlines()# U for univerial
raw[:75]
type(raw)
tokens = word_tokenize(str(raw))
len(raw)
type(tokens)

raw[:10]
tokens[:10]

text = nltk.Text(tokens)
text[1020:1060]

text.collocations()

words = [w.lower() for w in text]
vocab = sorted(set(words))
len(words)
len(vocab)
import re
[w for w in words if re.search('^[ghi][mno][jlk][def]$', w)]


text = nltk.word_tokenize("And now for something completely different")
nltk.pos_tag(text)

text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
x = nltk.pos_tag(text)
type(x)


text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
text.similar('woman')

frequency = nltk.defaultdict(int)
frequency['hello']
frequency['colorless']


pos = nltk.defaultdict(list)
pos['sleep'] = ['N','V']
pos['ideas']

last_letters = nltk.defaultdict(list)
words = nltk.corpus.words.words('en')
for word in words:
    key = word[-2:]
    last_letters[key].append(word)

last_letters['ng']


from nltk.corpus import names
import random
names = ([(name, 'male') for name in names.words('male.txt')] +
         [(name, 'female') for name in names.words('female.txt')])
random.shuffle(names)



def gender_features(word):
    return{'last_letter':word[-1]}

featuresets = [(gender_features(n), g) for (n,g) in names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier.classify(gender_features('Han'))
classifier.classify(gender_features('Meng'))
classifier.classify(gender_features('Gloria'))
classifier.classify(gender_features('Jessica'))

classifier.show_most_informative_features(5)

import nltk
from nltk.corpus import movie_reviews
print(movie_reviews.__sizeof__())
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
print(documents)
len(documents)
random.shuffle(documents)

type(documents)
len(documents)
documents[1]
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.keys()
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)'%word] = (word in document_words)
    return features

print(document_features(movie_reviews.words('pos/cv957_8737.txt')))

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

