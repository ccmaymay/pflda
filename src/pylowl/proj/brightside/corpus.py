import os
import re
import random
import math
import itertools as it

from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from concrete.communication.ttypes import Communication
from concrete.structure.ttypes import (
    SectionSegmentation, Section,
    SentenceSegmentation, Sentence,
    Tokenization, TokenList, Token,
    TokenLattice
)


SPLIT_RE = re.compile(r'[ :]')

SPECIAL_DOC_ATTRS = set(('id', 'text', 'tokens'))


def _pair_first_to_int(p):
    return (int(p[0]), p[1])


def load_vocab(filename):
    with open(filename) as f:
        vocab = dict(_pair_first_to_int(line.strip().split()) for line in f)
    return vocab


def doc_to_concrete_comm(doc):
    comm = Communication()
    comm.id = doc.id
    comm.text = doc.text

    comm.keyValueMap = dict()
    for (k, v) in doc.attrs.items():
        if k not in SPECIAL_DOC_ATTRS:
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


def write_concrete_docs(docs, output_dir):
    write_concrete_comms((doc_to_concrete_comm(doc) for doc in docs),
                         output_dir)


def write_concrete_comms(comms, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for (i, comm) in enumerate(comms):
        output_path = os.path.join(output_dir, '%d.concrete' % i)
        write_concrete_comm(comm, output_path)


def write_concrete_doc(doc, path):
    write_concrete_comm(doc_to_concrete_comm(doc), path)


def write_concrete_comm(comm, path):
    with open(path, 'wb') as f:
        transport = TTransport.TFileObjectTransport(f)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        comm.write(protocol)


def concrete_comm_to_doc(comm, section_segmentation_idx=0,
                         sentence_segmentation_idx=0, tokenization_list_idx=0):
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

    return Document(tokens, id=comm.id, text=comm.text, **attrs)


def load_concrete_docs(paths, section_segmentation_idx=0,
                       sentence_segmentation_idx=0, tokenization_list_idx=0):
    for (comm, path) in load_concrete_comms(paths):
        doc = concrete_comm_to_doc(comm, section_segmentation_idx,
                                   sentence_segmentation_idx,
                                   tokenization_list_idx)
        doc.path = path
        yield doc


def load_concrete_comms(paths):
    for input_path in paths:
        yield (load_concrete_comm(input_path), input_path)


def load_concrete_doc(path, section_segmentation_idx=0,
                      sentence_segmentation_idx=0, tokenization_list_idx=0):
    return concrete_comm_to_doc(load_concrete_comm(path),
                                section_segmentation_idx,
                                sentence_segmentation_idx,
                                tokenization_list_idx)


def load_concrete_comm(path):
    comm = Communication()
    with open(path, 'rb') as f:
        transportIn = TTransport.TFileObjectTransport(f)
        protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
        comm.read(protocolIn)
    return comm


def update_concrete_comms(paths, transform):
    for (comm, path) in load_concrete_comm(paths):
        transform(comm)
        write_concrete_comm(comm, path)


def load_lattice_best_tokens(path):
    with open(path, 'rb') as f:
        transportIn = TTransport.TFileObjectTransport(f)
        protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
        lattice = TokenLattice()
        lattice.read(protocolIn)
        best_path = lattice.cachedBestPath
        if best_path is None:
            return None
        else:
            token_list = best_path.tokenList
            if token_list is None:
                return None
            else:
                return [t.text for t in token_list]


class Document(object):
    '''
    A single document: a list of type-count pairs and an optional
    id (e.g., filename and line number).
    '''

    def __init__(self, tokens, id=None, text=None, **attrs):
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

        self.id = id
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
            id=self.id, text=self.text, **self.attrs)
        # may be empty
        test_doc = Document(tokens[num_train_tokens:],
            id=self.id, text=self.text, **self.attrs)
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
        concrete_docs = load_concrete_docs(paths,
                                      section_segmentation,
                                      sentence_segmentation,
                                      tokenization_list)
        docs = [Document([r_vocab[t] for t in doc.tokens],
                         id=doc.id, text=doc.text, **doc.attrs)
                for doc in concrete_docs]
        return Corpus(docs, len(paths))

    @classmethod
    def from_concrete_stream(cls, paths, r_vocab, num_docs, section_segmentation=0, sentence_segmentation=0, tokenization_list=0):
        '''
        Return corpus containing at most num_docs documents from a list
        of concrete document paths.  Documents are loaded lazily.
        '''
        # TODO: pass file-like object(s) instead of paths
        concrete_docs = load_concrete_docs(paths,
                                      section_segmentation,
                                      sentence_segmentation,
                                      tokenization_list)
        docs = (Document([r_vocab[t] for t in doc.tokens],
                         id=doc.id, text=doc.text, **doc.attrs)
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
