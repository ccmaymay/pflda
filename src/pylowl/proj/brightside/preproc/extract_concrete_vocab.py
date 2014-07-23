#!/usr/bin/env python


from pylowl.proj.brightside.utils import nested_file_paths, mkdirp_parent
from pylowl.proj.brightside.preproc.utils import write_vocab
from pylowl.proj.brightside.corpus import load_concrete_docs


def extract_concrete_vocab(input_paths, section_segmentation,
                           sentence_segmentation, tokenization_list,
                           vocab_output_path):
    mkdirp_parent(vocab_output_path)

    concrete_docs = load_concrete_docs(input_paths,
                                  int(section_segmentation),
                                  int(sentence_segmentation),
                                  int(tokenization_list))
    vocab = dict()
    for doc in concrete_docs:
        for word in doc.words:
            if word not in vocab:
                vocab[word] = len(vocab)
    write_vocab(vocab_output_path, vocab)


if __name__ == '__main__':
    import sys
    paths = nested_file_paths(sys.argv[1])
    section_segmentation = int(sys.argv[2])
    sentence_segmentation = int(sys.argv[3])
    tokenization_list = int(sys.argv[4])
    vocab_output_path = sys.argv[5]
    extract_concrete_vocab(paths, section_segmentation,
                           sentence_segmentation, tokenization_list,
                           vocab_output_path)
