#!/usr/bin/env python


import os
import re
import tempfile
from pylowl.proj.brightside.corpus import write_concrete_doc, Document
from pylowl.proj.brightside.utils import nested_file_paths
from pylowl.proj.brightside.preproc.extract_concrete_vocab import extract_concrete_vocab


SPLIT_RE = re.compile(r'\W+')
SRC_EXTENSIONS = ('.py', '.sh', '.c', '.h', '.pxd', '.pyx')


def src_path_filter(path):
    ext = path[path.rfind('.'):]
    return (ext in SRC_EXTENSIONS
            and not (ext == '.c' and 'src/pylowl/' in path))


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def make_data(input_dir, output_dir):
    i = 0
    for path in nested_file_paths(input_dir):
        if src_path_filter(path):
            tokens = []
            with open(path) as f:
                for line in f:
                    tokens.extend(token for token in SPLIT_RE.split(line)
                                  if token and not is_num(token))
            if tokens:
                output_path = os.path.join(output_dir, '%d.concrete' % i)
                ext = os.path.splitext(path)[1]
                attrs = {'class': ext, 'user': ext}
                write_concrete_doc(Document(tokens, id=path, **attrs),
                                   output_path)
                i += 1


def src_to_concrete(src_dir):
    data_dir = tempfile.mkdtemp()
    make_data(src_dir, data_dir)
    train_data_dir = data_dir
    # TODO would be nice if we didn't test on training data...
    test_data_dir = data_dir

    (fd, vocab_path) = tempfile.mkstemp()
    os.close(fd)
    extract_concrete_vocab(nested_file_paths(train_data_dir), 0, 0, 0, vocab_path)

    return (train_data_dir, test_data_dir, vocab_path)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(src_dir='src')
    parser.add_argument('--src_dir', type=str, required=False,
                        help='path to littleowl source tree')
    args = parser.parse_args()

    (train_data_dir, test_data_dir, vocab_path) = src_to_concrete(args.src_dir)

    print ' '.join((train_data_dir, test_data_dir, vocab_path))
