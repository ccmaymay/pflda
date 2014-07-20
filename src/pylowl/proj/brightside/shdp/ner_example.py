#!/usr/bin/env python

import nltk

text = "We'll go to Texas or Southern California tonight."
tok_sent = nltk.word_tokenize(text)
pos_sent = nltk.pos_tag(tok_sent)
ner_sent = nltk.ne_chunk(pos_sent)
for tree in ner_sent.subtrees():
    if tree.node == 'GPE':
        print tree
