#!/usr/bin/env python


import os
from utils import make_parent_dir


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(doc_ids=False)
    parser.add_argument('input_path', type=str,
                        help='rasp-processed input directory')
    parser.add_argument('output_path', type=str,
                        help='output file path')
    parser.add_argument('--doc_ids', action='store_true',
                        help='insert (orig) doc ID at beginning of each line')
    args = parser.parse_args()
    format_to_doc_per_line(args.input_path, args.output_path, args.doc_ids)


def format_to_doc_per_line(input_path, output_path, doc_ids=False):
    if not os.path.isdir(input_path):
        raise Exception('"%s" does not seem to be a directory' % input_path)
    make_parent_dir(output_path)
    with open(output_path, 'w') as out_f:
        for (dir_path, dir_entries, file_entries) in os.walk(input_path):
            for filename in file_entries:
                input_file_path = os.path.join(dir_path, filename)
                with open(input_file_path) as f:
                    if doc_ids:
                        out_f.write(input_file_path)
                    i = 0
                    for line in f:
                        if doc_ids or i > 0:
                            out_f.write(' ')
                        out_f.write(line.strip())
                        i += 1
                    out_f.write('\n')


if __name__ == '__main__':
    main()
