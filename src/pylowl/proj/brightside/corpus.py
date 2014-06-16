import re
import random
import math
import itertools as it
from utils import path_list


SPLIT_RE = re.compile(r'[ :]')


class Document(object):
    '''
    A single document: a list of type-count pairs and an optional
    identifier (e.g., filename and line number).
    '''

    def __init__(self, type_ids, type_counts, identifier=None):
        self.words = type_ids
        self.length = len(self.words)
        self.counts = type_counts
        self.total = sum(self.counts)
        self.identifier = identifier

    def split(self, train_frac):
        '''
        Return pair of documents, the first containing roughly
        train_frac (a fraction) of this document's tokens
        (chosen uniformly at random), and the second containing
        the rest.  As long as this document (self) is non-empty,
        the first ("training") document is always non-empty, while
        the second ("testing") document may be empty.
        '''
        tokens = []
        for (w, c) in it.izip(self.words, self.counts):
            tokens += [w] * c
        random.shuffle(tokens)
        num_train_tokens = int(math.ceil(len(tokens) * train_frac))
        # always at least one token in training document
        train_doc = Document.from_tokens(tokens[:num_train_tokens],
            self.identifier)
        # may be empty
        test_doc = Document.from_tokens(tokens[num_train_tokens:],
            self.identifier)
        return (train_doc, test_doc)

    @classmethod
    def from_tokens(cls, tokens, identifier=None):
        '''
        Return document from list of tokens (where each token is an
        integral type index).
        '''
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
        '''
        Return document specified by line, which should be an integer
        followed by a list of integer pairs, delimited by whitespace,
        where the two integers within each pair are separated by a
        colon, e.g.:
            3 2:3 4:5 5:3
        The first integer is the number of types in the document;
        each pair after that contains a type index (first) and a count
        (second).
        '''
        pieces = [int(i) for i in SPLIT_RE.split(line)]
        type_ids = pieces[1::2]
        type_counts = pieces[2::2]
        return Document(type_ids, type_counts, identifier)


class Corpus(object):
    '''
    Container for a list of documents.  Documents may be loaded unlazily
    or lazily, facilitating streaming processing.
    '''

    def __init__(self, docs, num_docs):
        self.docs = docs
        self.num_docs = num_docs

    @classmethod
    def from_data(cls, loc):
        '''
        Return corpus containing all documents from a path or
        list of paths.  Documents are loaded immediately (unlazily).
        '''
        docs = []
        i = 0
        for path in path_list(loc):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(Document.from_line(line,
                            identifier=':'.join((path, str(i)))))
                    i += 1
        return Corpus(docs, len(docs))

    @classmethod
    def from_stream_data(cls, f, num_docs):
        '''
        Return corpus containing at most num_docs documents from the
        file-like object f.  Documents are loaded lazily.
        '''
        docs = (Document.from_line(line.strip(), identifier=str(i))
                for (i, line) in enumerate(f)
                if i < num_docs)
        return Corpus(docs, num_docs) # TODO what if too short?

    def split_within_docs(self, train_frac):
        '''
        Return pair of corpora, both with the same number of documents
        as this corpus (self).  The documents in the first corpus
        will contain roughly train_frac (a fraction) of the tokens in
        the respective documents in this corpus, chosen uniformly at
        random; the documents in the second corpus will contain the
        rest.  Assuming documents in this corpus (self) are non-empty,
        documents in the first returned corpus are always non-empty
        while documents in the second may be empty.
        '''
        doc_pairs = [p for p in
                     (d.split(train_frac) for d in self.docs)
                     if p[0].words and p[1].words]
        docs_train = [p[0] for p in doc_pairs]
        docs_test = [p[1] for p in doc_pairs]
        return (Corpus(docs_train, self.num_docs),
                Corpus(docs_test, self.num_docs))
