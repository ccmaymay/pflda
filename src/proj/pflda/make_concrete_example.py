#!/usr/bin/env python


from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol

from concrete.communication import *
from concrete.communication.ttypes import *
from concrete.structure.ttypes import *

from data import Dataset

import os


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
    section.sentenceSegmentation = [sentenceSegmentation] # TODO typo?
    sectionSegmentation.sectionList = [section]
    comm.sectionSegmentations = [sectionSegmentation]
    return comm


def main(output_dir, data_dir, categories):
    os.mkdir(output_dir)
    dataset = Dataset(data_dir, categories)
    for it in (dataset.train_iterator, dataset.test_iterator):
        for (doc_idx, category, tokens) in it():
            comm = make_comm(tokens)
            output_filename = os.path.join(output_dir, '%d.dat' % doc_idx)
            with open(output_filename, 'wb') as out_f:
                transport = TTransport.TFileObjectTransport(out_f)
                protocol = TBinaryProtocol.TBinaryProtocol(transport)
                comm.write(protocol)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 4:
        raise Exception('Specify output dir, data dir, and categories.\n')

    output_dir = sys.argv[1]
    data_dir = sys.argv[2]
    categories = set(sys.argv[3].split(','))
    main(output_dir, data_dir, categories)
