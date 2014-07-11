import os
import re
import random
import math
import itertools as it
from utils import path_list


SPLIT_RE = re.compile(r'[ :]')


def _pair_first_to_int(p):
    return (int(p[0]), p[1])


def load_vocab(filename):
    with open(filename) as f:
        vocab = dict(_pair_first_to_int(line.strip().split()) for line in f)
    return vocab


def write_concrete(docs, output_dir):
    from thrift.transport import TTransport
    from thrift.protocol import TBinaryProtocol
    from concrete.communication.ttypes import Communication
    from concrete.structure.ttypes import (
        SectionSegmentation, Section,
        SentenceSegmentation, Sentence,
        Tokenization, TokenList, Token
    )

    SPECIAL_KEYS = set(('text', 'tokens'))

    def make_comm(doc):
        comm = Communication()
        comm.text = doc.text

        comm.keyValueMap = dict()
        for (k, v) in doc.attrs.items():
            if k not in SPECIAL_KEYS:
                comm.keyValueMap[k] = v

        sectionSegmentation = SectionSegmentation()
        section = Section()
        sentenceSegmentation = SentenceSegmentation()
        sentence = Sentence()
        tokenization = Tokenization()
        tokenization.tokenList = TokenList([Token(text=t) for t in doc.tokens])
        sentence.tokenizationList = [tokenization]
        sentenceSegmentation.sentenceList = [sentence]
        section.sentenceSegmentation = [sentenceSegmentation]
        sectionSegmentation.sectionList = [section]
        comm.sectionSegmentations = [sectionSegmentation]

        return comm

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for (i, doc) in enumerate(docs):
        comm = make_comm(doc)
        output_path = os.path.join(output_dir, '%d.concrete' % i)
        with open(output_path, 'wb') as f:
            transport = TTransport.TFileObjectTransport(f)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            comm.write(protocol)


def load_concrete(loc, section_segmentation_idx=0, sentence_segmentation_idx=0,
                  tokenization_list_idx=0):
    from thrift.transport import TTransport
    from thrift.protocol import TBinaryProtocol
    from concrete.communication.ttypes import Communication

    def parse_comm(comm):
        tokens = []
        if comm.sectionSegmentations is not None:
            section_segmentation = comm.sectionSegmentations[section_segmentation_idx]
            if section_segmentation.sectionList is not None:
                for section in section_segmentation.sectionList:
                    if section.sentenceSegmentation is not None:
                        sentence_segmentation = section.sentenceSegmentation[sentence_segmentation_idx]
                        if sentence_segmentation.sentenceList is not None:
                            for sentence in sentence_segmentation.sentenceList:
                                if sentence.tokenizationList is not None:
                                    tokenization = sentence.tokenizationList[tokenization_list_idx]
                                    if tokenization.tokenList is not None:
                                        for token in tokenization.tokenList.tokens:
                                            tokens.append(token.text)

        if comm.keyValueMap is None:
            attrs = dict()
        else:
            attrs = comm.keyValueMap

        return Document(tokens, text=comm.text, **attrs)

    for input_path in path_list(loc):
        with open(input_path, 'rb') as f:
            transportIn = TTransport.TFileObjectTransport(f)
            protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
            comm = Communication()
            comm.read(protocolIn)
            yield parse_comm(comm)


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
            text=self.text, **self.attrs)
        # may be empty
        test_doc = Document(tokens[num_train_tokens:],
            text=self.text, **self.attrs)
        return (train_doc, test_doc)

    def __str__(self):
        return ' '.join(self.tokens)


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
        docs = [Document([r_vocab[t] for t in doc.tokens],
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
        docs = (Document([r_vocab[t] for t in doc.tokens],
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
