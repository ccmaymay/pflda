#!/usr/bin/env python


from pylowl.proj.brightside.preproc.utils import write_vocab, make_parent_dir, nested_file_paths
from pylowl.proj.brightside.corpus import load_concrete_docs


def extract_concrete_vocab(input_paths, section_segmentation,
                           sentence_segmentation, tokenization_list,
                           vocab_output_path):
    make_parent_dir(vocab_output_path)

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
    paths = [path for d in sys.argv[1:] for path in nested_file_paths(d)]
    extract_concrete_vocab(paths, section_segmentation,
                           sentence_segmentation, tokenization_list,
                           vocab_output_path)
