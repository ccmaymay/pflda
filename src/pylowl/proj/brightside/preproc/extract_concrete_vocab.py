#!/usr/bin/env python


from glob import glob
from utils import write_vocab, make_parent_dir
from pylowl.proj.brightside.utils import load_concrete


def main(input_pattern, section_segmentation, sentence_segmentation,
         tokenization_list, vocab_output_path):
    make_parent_dir(vocab_output_path)

    concrete_docs = load_concrete(glob(input_pattern),
                                  int(section_segmentation),
                                  int(sentence_segmentation),
                                  int(tokenization_list))
    vocab = dict()
    for (path, doc) in concrete_docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    write_vocab(vocab_output_path, vocab)


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])