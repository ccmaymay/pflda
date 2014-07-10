import re
import random
import math
import itertools as it
from utils import path_list, load_concrete


SPLIT_RE = re.compile(r'[ :]')


class Document(object):
    '''
    A single document: a list of type-count pairs and an optional
    identifier (e.g., filename and line number).
    '''

    def __init__(self, tokens, text=None, **attrs):
        self.tokens = tokens

        self.words = []
        self.counts = []

        types = dict()
        for token in self.tokens:
            if token in types:
                self.counts[types[token]] += 1
            else:
                types[token] = len(self.words)
                self.words.append(token)
                self.counts.append(1)

        self.length = len(self.words)
        self.total = sum(self.counts)

        self.text = text
        self.attrs = attrs

    def split(self, train_frac):
        '''
        Return pair of documents, the first containing roughly
        train_frac (a fraction) of this document's tokens
        (chosen uniformly at random), and the second containing
        the rest.  As long as this document (self) is non-empty,
        the first ("training") document is always non-empty, while
        the second ("testing") document may be empty.
        '''
        tokens = [t for t in self.tokens]
        random.shuffle(tokens)

        num_train_tokens = int(math.ceil(len(tokens) * train_frac))
        # always at least one token in training document
        train_doc = Document(tokens[:num_train_tokens],
            text=self.text, **attrs)
        # may be empty
        test_doc = Document(tokens[num_train_tokens:],
            text=self.text, **attrs)
        return (train_doc, test_doc)


class Corpus(object):
    '''
    Container for a list of documents.  Documents may be loaded unlazily
    or lazily, facilitating streaming processing.
    '''

    def __init__(self, docs, num_docs):
        self.docs = docs
        self.num_docs = num_docs

    @classmethod
    def from_concrete(cls, paths, r_vocab, section_segmentation=0, sentence_segmentation=0, tokenization_list=0):
        '''
        Return corpus containing all documents from a list of concrete
        document paths.  Documents are loaded immediately (unlazily).
        '''
        concrete_docs = load_concrete(paths,
                                      section_segmentation,
                                      sentence_segmentation,
                                      tokenization_list)
        docs = [Document([r_vocab[t] for t in tokens],
                         text=doc.text, **doc.attrs)
                for doc in concrete_docs]
        return Corpus(docs, len(paths))

    @classmethod
    def from_concrete_stream(cls, paths, r_vocab, num_docs, section_segmentation=0, sentence_segmentation=0, tokenization_list=0):
        '''
        Return corpus containing at most num_docs documents from a list
        of concrete document paths.  Documents are loaded lazily.
        '''
        # TODO: pass file-like object(s) instead of paths
        concrete_docs = load_concrete(paths,
                                      section_segmentation,
                                      sentence_segmentation,
                                      tokenization_list)
        docs = (Document([r_vocab[t] for t in tokens],
                         text=doc.text, **doc.attrs)
                for (i, doc) in enumerate(concrete_docs)
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
