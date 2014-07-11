#!/usr/bin/env python


import math
import os
import random
from glob import glob
from pylowl.proj.brightside.corpus import load_concrete_raw, write_concrete_raw, Document


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(train_frac=0.6)
    parser.add_argument('input_path', type=str,
                        help='doc-per-line input directory path')
    parser.add_argument('output_path', type=str,
                        help='doc-per-line output directory path')
    parser.add_argument('--train_frac', type=float,
                        help='fraction of dataset allocated for training')

    args = parser.parse_args()
    split_train_test(
        args.input_path,
        args.output_path,
        args.train_frac,
    )


def iter_docs(input_path):
    with open(input_path) as f:
        for (line_num, line) in enumerate(f):
            pieces = line.strip().split('\t')
            user = pieces[0]
            datetime = pieces[1]
            latitude = pieces[3]
            longitude = pieces[3]
            text = '\t'.join(pieces[5:])
            tokens = [t for t in text.split()
                      if not [c for c in t if ord(c) > 127]]
            text = ' '.join(tokens)
            yield Document(tokens,
                text=text,
                user=user,
                datetime=datetime,
                latitude=latitude,
                longitude=longitude,
                identifier=str(line_num),
            )


def split_train_test(input_path, output_path, train_frac):
    loc = glob(input_path)

    num_docs = len(loc)
    doc_ids = range(num_docs)
    random.shuffle(doc_ids)
    doc_ids_split = int(math.ceil(train_frac * num_docs))
    train_doc_ids = set(doc_ids[:doc_ids_split])

    write_concrete_raw(
        (doc for (i, doc) in enumerate(load_concrete_raw(loc))
         if i in train_doc_ids),
        os.path.join(output_path, 'train'))
    write_concrete_raw(
        (doc for (i, doc) in enumerate(load_concrete_raw(loc))
         if i not in train_doc_ids),
        os.path.join(output_path, 'test'))


if __name__ == '__main__':
    main()
