#!/usr/bin/env python


import math
import os
import random
from glob import glob
from pylowl.proj.brightside.corpus import load_concrete_raw, write_concrete_raw, Document


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(train_frac=0.6, shuffle=False)
    parser.add_argument('input_path', type=str,
                        help='doc-per-line input directory path')
    parser.add_argument('output_path', type=str,
                        help='doc-per-line output directory path')
    parser.add_argument('--train_frac', type=float,
                        help='fraction of dataset allocated for training')
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle docs before split')

    args = parser.parse_args()
    split_train_test(
        args.input_path,
        args.output_path,
        args.train_frac,
        args.shuffle,
    )


def split_train_test(input_path, output_path, train_frac, shuffle):
    loc = glob(input_path)

    num_docs = len(loc)
    doc_indices = range(num_docs)
    if shuffle:
        random.shuffle(doc_indices)
    doc_indices_split = int(math.ceil(train_frac * num_docs))
    train_doc_indices = set(doc_indices[:doc_indices_split])

    write_concrete_raw(
        (comm for (i, (comm, path)) in enumerate(load_concrete_raw(loc))
         if i in train_doc_indices),
        os.path.join(output_path, 'train'))
    write_concrete_raw(
        (comm for (i, (comm, path)) in enumerate(load_concrete_raw(loc))
         if i not in train_doc_indices),
        os.path.join(output_path, 'test'))


if __name__ == '__main__':
    main()
