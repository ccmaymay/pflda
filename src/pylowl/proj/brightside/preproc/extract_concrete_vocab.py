#!/usr/bin/env python


from glob import glob
from utils import write_vocab, make_parent_dir
from pylowl.proj.brightside.utils import load_concrete


def extract_concrete_vocab(input_paths, section_segmentation,
                           sentence_segmentation, tokenization_list,
                           vocab_output_path):
    make_parent_dir(vocab_output_path)

    concrete_docs = load_concrete(input_paths,
                                  int(section_segmentation),
                                  int(sentence_segmentation),
                                  int(tokenization_list))
    vocab = dict()
    for doc in concrete_docs:
        for word in doc.words:
            if word not in vocab:
                vocab[word] = len(vocab)
    write_vocab(vocab_output_path, vocab)


def main(input_pattern, section_segmentation,
         sentence_segmentation, tokenization_list,
         vocab_output_path):
    extract_concrete_vocab(glob(input_pattern), section_segmentation,
                           sentence_segmentation, tokenization_list,
                           vocab_output_path)


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
