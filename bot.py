#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import itertools
from collections import Counter
import h5py
import numpy as np
from nltk.tokenize import word_tokenize

filepath = 'data/'
topN = 50000
seq_length = 23 # 25 finally
f = h5py.File(filepath+"bot.hdf5","w")

def buildCorpus(filename="wikipedia.talkpages.conversations.txt"):
	filename = filepath + filename
	sents = []
	with open(filename,'r') as f:
		for line in f.readlines():
			if len(line) > 60:
				_t = line.split("+++$+++")
				# print(line)
				dialogue = _t[7].strip("")
				sent = word_tokenize(dialogue)
				sent = [s.lower() for s in sent]
				if len(sent) < seq_length:
					sent += (seq_length-len(sent)) * ['<blank>']
				else:
					sent = sent[:seq_length]
				sent = ['<bos>'] + sent + ['<eos>']
				sents.append(sent)
	return sents

def buildVocab(sents):
	word_counts = Counter(itertools.chain(*sents))
	vocabulary_inv = [x[0] for x in word_counts.most_common()[:topN-1]]
	vocabulary = {}
	vocabulary = {x: i+2 for i, x in enumerate(vocabulary_inv)} 
	vocabulary["<unk>"] = 1
	vocabulary_txt = sorted(vocabulary.iteritems(), key=lambda d:d[1], reverse = False)
	vf = file(filepath+"bot.dict","w")
	for word,_ in vocabulary_txt:
		vf.write(word)
		vf.write("\n")
	vf.close()
	return vocabulary # dict

def buildSents(_sents,vocabulary):
	sources = []
	targets = []
	for i,_sent in enumerate(_sents):
		sent = [vocabulary[w] if w in vocabulary else vocabulary['<unk>'] for w in _sent]
		if i % 2 == 0:
			sources.append(sent)
		else:
			targets.append(sent)
	return sources,targets

if __name__ == "__main__":
	print("building corpus...")
	sents = buildCorpus()
	# print(sents)
	print("building vocab...")
	vocabulary = buildVocab(sents)
	print("building sents...")
	sources,targets = buildSents(sents,vocabulary)
	count = min(len(sources),len(targets))
	print("totally %d sents in sources and targets" %(count))
	print("saving to bot.hdf5....")
	sources = sources[:count]
	targets = targets[:count]
	f["source"] = np.array(sources)
	f["target"] = np.array(targets)
	f.close()






	






	
	