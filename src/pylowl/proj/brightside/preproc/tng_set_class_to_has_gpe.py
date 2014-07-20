#!/usr/bin/env python


import nltk
from pylowl.proj.brightside.utils import nested_file_paths
from pylowl.proj.brightside.corpus import update_concrete_comms


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', type=str,
                        help='data directory path')

    args = parser.parse_args()
    tng_set_class_to_has_gpe(args.data_dir)


def set_class(comm):
    has_gpe = False
    for sentence in nltk.sent_tokenize(comm.text):
        sentence = nltk.word_tokenize(sentence)
        sentence = nltk.pos_tag(sentence)
        sentence = nltk.ne_chunk(sentence)
        for tree in sentence.subtrees():
            if tree.node == 'GPE':
                has_gpe = True
    comm.attrs['class'] = 'has_gpe' if has_gpe else 'no_gpe'


def tng_set_class_to_has_gpe(data_dir):
    update_concrete_comms(nested_file_paths(data_dir), set_class)


if __name__ == '__main__':
    main()
