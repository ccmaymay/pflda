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

    def __init__(self, docs, num_docs):
        self.docs = docs
        self.num_docs = len(self.docs)

    @classmethod
    def from_data(cls, filename):
        docs = []
        i = 0
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(Document.from_line(line,
                        identifier=':'.join((filename, str(i)))))
                i += 1
        return Corpus(docs, len(docs))

    @classmethod
    def from_stream_data(cls, f, num_docs):
        lines = (line.strip() for line in f)
        non_empty_lines = ((j, line) for (j, line) in enumerate(lines) if line)
        docs = (Document.from_line(line, identifier=str(j))
                for (i, line) in enumerate(non_empty_lines)
                if i < num_docs)
        return Corpus(docs, num_docs) # TODO what if too short?

    def split_within_docs(self, train_frac):
        doc_pairs = (p for p in
                     (d.split(train_frac) for d in self.docs)
                     if p[0].words and p[1].words)
        docs_train = (p[0] for p in doc_pairs)
        docs_test = (p[1] for p in doc_pairs)
        c_train = Corpus(docs_train, self.num_docs) # TODO what if too short?
        c_test = Corpus(docs_test, self.num_docs) # TODO what if too short?
        return (c_train, c_test)
