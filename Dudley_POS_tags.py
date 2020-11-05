# A simple POS tagger that also uses the Dickens.txt file to list and plot the most common nouns in a given corpus.

import nltk, numpy, matplotlib, argparse
from timeit import default_timer as timer
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", action="store", default="dickens.txt", help="Corpus to generate from")
options = parser.parse_args()
file = options.file

def getTags():
    text = open(file, 'r', encoding="utf8").read()
    tokenized = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokenized)
    return tagged

def getNouns():
    tagged = getTags()
    nouns_dict = {}
    for word, tag in tagged:
        if tag in ["NN", "NNP", "NNPS", "NNS"]:
            if word not in nouns_dict:
                nouns_dict[word] = 0
            nouns_dict[word] += 1
    return nouns_dict

def plotNouns():
    nouns_dict = getNouns()
    count = Counter(nouns_dict)
    high = count.most_common(
        10)  # https://www.geeksforgeeks.org/python-program-to-find-the-highest-3-values-in-a-dictionary/
    print(high)
    word_dist = nltk.FreqDist(nouns_dict)
    word_dist.plot(100)

start = timer()
plotNouns()
print("Plotting Nouns:", timer()-start)
