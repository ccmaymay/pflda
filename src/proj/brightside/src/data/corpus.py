'''
3 2:3 4:5 5:3 --- document info (word: count)
'''


import re
import random
import math
import itertools as it


SPLIT_RE = re.compile(r'[ :]')


class Document(object):

    ''' the class for a single document '''

    def __init__(self, type_ids, type_counts, identifier=None):
        self.words = type_ids
        self.length = len(self.words)
        self.counts = type_counts
        self.total = sum(self.counts)
        self.identifier = identifier

    def split(self, train_frac):
        tokens = []
        for (w, c) in it.izip(self.words, self.counts):
            tokens += [w] * c
        random.shuffle(tokens)
        num_train_tokens = int(math.ceil(len(tokens) * train_frac))
        train_doc = Document.from_tokens(tokens[:num_train_tokens],
            self.identifier)
        test_doc = Document.from_tokens(tokens[num_train_tokens:],
            self.identifier)
        return (train_doc, test_doc)

    @classmethod
    def from_tokens(cls, tokens, identifier=None):
        types_dict = dict()
        for t in tokens:
            if t in types_dict:
                types_dict[t] += 1
            else:
                types_dict[t] = 1

        type_ids = []
        type_counts = []
        for (type_id, type_count) in types_dict.items():
            type_ids.append(type_id)
            type_counts.append(type_count)

        return Document(type_ids, type_counts, identifier)

    @classmethod
    def from_line(cls, line, identifier=None):
        pieces = [int(i) for i in SPLIT_RE.split(line)]
        type_ids = pieces[1::2]
        type_counts = pieces[2::2]
        return Document(type_ids, type_counts, identifier)


class Corpus(object):

    ''' the class for the whole corpus'''

    def __init__(self, docs, vocab_size=None):
        if vocab_size is None:
            self.vocab_size = 0
            for d in docs:
                if d.length > 0:
                    max_word = max(d.words)
                    if max_word >= self.vocab_size:
                        self.vocab_size = max_word + 1
        else:
            self.vocab_size = vocab_size

        self.docs = docs
        self.num_docs = len(self.docs)

    @classmethod
    def from_data(cls, filename):
        docs = []
        i = 0
        for line in open(filename):
            docs.append(Document.from_line(line,
                identifier=':'.join((filename, str(i)))))
            i += 1
        return Corpus(docs)

    @classmethod
    def from_stream_data(cls, f, num_docs):
        docs = []
        for i in xrange(num_docs):
            line = f.readline()
            line = line.strip()
            if len(line) == 0:
                break
            docs.append(Document.from_line(line, identifier=str(i)))
        return Corpus(docs)

    def split_within_docs(self, train_frac):
        doc_pairs = [p for p in
                     (d.split(train_frac) for d in self.docs)
                     if p[0].words and p[1].words]
        docs_train = [p[0] for p in doc_pairs]
        docs_test = [p[1] for p in doc_pairs]
        c_train = Corpus(docs_train, self.vocab_size)
        c_test = Corpus(docs_test, self.vocab_size)
        return (c_train, c_test)
