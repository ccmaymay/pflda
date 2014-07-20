#!/usr/bin/env python


import os
from pylowl.proj.brightside.utils import nested_file_paths
from pylowl.proj.brightside.corpus import load_concrete_docs


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', type=str,
                        help='input directory path')
    parser.add_argument('output_dir', type=str,
                        help='output directory path')
    args = parser.parse_args()
    concrete_to_text(
        args.input_dir,
        args.output_dir,
    )


def concrete_to_text(input_dir, output_dir):
    input_paths = nested_file_paths(input_dir)
    os.makedirs(output_dir)
    i = 0
    for doc in load_concrete_docs(input_paths):
        path = os.path.join(output_dir, '%d.txt' % i)
        with open(path, 'w') as f:
            f.write(doc.text)


if __name__ == '__main__':
    main()
