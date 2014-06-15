#!/usr/bin/env python


import os
import itertools as it
from data import write_vocab, write_doc
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from concrete.communication.ttypes import Communication


def parse_comm(comm):
    tokens = []
    if comm.sectionSegmentations is not None:
        for sect_seg in comm.sectionSegmentations:
            if sect_seg.sectionList is not None:
                for sect in sect_seg.sectionList:
                    if sect.sentenceSegmentation is not None:
                        for sent_seg in sect.sentenceSegmentation:
                            if sent_seg.sentenceList is not None:
                                for sent in sent_seg.sentenceList:
                                    if sent.tokenizationList is not None:
                                        for tokzn in sent.tokenizationList:
                                            if tokzn.tokenList is not None:
                                                for tok in tokzn.tokenList:
                                                    tokens.append(tok.text)
    return tokens


def load_dir(input_dir):
    for input_filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, input_filename)
        if os.path.isfile(input_path):
            with open(input_path, 'rb') as f:
                transportIn = TTransport.TFileObjectTransport(f)
                protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
                comm = Communication()
                comm.read(protocolIn)
                tokens = parse_comm(comm)
                yield tokens


def main(train_input_dir, test_input_dir, output_dir):
    os.makedirs(output_dir)

    train_output_path = os.path.join(output_dir, 'train')
    test_output_path = os.path.join(output_dir, 'test')
    vocab_output_path = os.path.join(output_dir, 'vocab')

    vocab = dict()
    for doc in it.chain(load_dir(train_input_dir), load_dir(test_input_dir)):
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)

    write_vocab(vocab_output_path, vocab)

    with open(train_output_path, 'w') as f:
        for doc in load_dir(train_input_dir):
            write_doc(f, doc, vocab)

    with open(test_output_path, 'w') as f:
        for doc in load_dir(test_input_dir):
            write_doc(f, doc, vocab)


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
