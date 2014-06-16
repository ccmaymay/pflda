#!/usr/bin/env python


from glob import glob
from utils import write_vocab, make_parent_dir
from pylowl.proj.brightside.utils import load_concrete


def main(input_pattern, vocab_output_path):
    make_parent_dir(vocab_output_path)

    vocab = dict()
    for doc in load_concrete(glob(input_pattern)):
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    write_vocab(vocab_output_path, vocab)


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
