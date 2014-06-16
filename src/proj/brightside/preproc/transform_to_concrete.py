#!/usr/bin/env python


from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from concrete.communication.ttypes import Communication
from concrete.structure.ttypes import (
    SectionSegmentation, Section,
    SentenceSegmentation, Sentence,
    Tokenization, Token
)
import os
from utils import load_vocab
from corpus import Corpus


def make_comm(tokens):
    comm = Communication()
    comm.text = ' '.join(tokens)
    sectionSegmentation = SectionSegmentation()
    section = Section()
    sentenceSegmentation = SentenceSegmentation()
    sentence = Sentence()
    tokenization = Tokenization()
    tokenization.tokenList = [Token(text=t) for t in tokens]
    sentence.tokenizationList = [tokenization]
    sentenceSegmentation.sentenceList = [sentence]
    section.sentenceSegmentation = [sentenceSegmentation]
    sectionSegmentation.sectionList = [section]
    comm.sectionSegmentations = [sectionSegmentation]
    return comm


def write_docs(input_path, output_dir, vocab):
    os.makedirs(output_dir)

    corpus = Corpus.from_data(input_path)
    for (i, doc) in enumerate(corpus.docs):
        tokens = []
        for (word, count) in zip(doc.words, doc.counts):
            tokens.extend([vocab[word]] * count)
        comm = make_comm(tokens)

        output_path = os.path.join(output_dir, '%d.dat' % i)
        with open(output_path, 'wb') as f:
            transport = TTransport.TFileObjectTransport(f)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            comm.write(protocol)


def main(train_input_path, test_input_path, vocab_input_path, output_dir):
    vocab = load_vocab(vocab_input_path)

    train_output_dir = os.path.join(output_dir, 'train')
    test_output_dir = os.path.join(output_dir, 'test')

    write_docs(train_input_path, train_output_dir, vocab)
    write_docs(test_input_path, test_output_dir, vocab)


if __name__ == '__main__':
    import sys

    args = []
    params = dict()
    for token in sys.argv[1:]:
        eq_pos = token.find('=')
        if token.startswith('--') and eq_pos >= 0:
            k = token[len('--'):eq_pos]
            v = token[(eq_pos+1):len(token)]
            params[k] = v
        else:
            args.append(token)

    main(*args, **params)
