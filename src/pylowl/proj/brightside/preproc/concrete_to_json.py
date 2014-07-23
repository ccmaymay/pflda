#!/usr/bin/env python


import os
import json
import codecs
from pylowl.proj.brightside.utils import nested_input_output_file_paths, path_is_concrete
from pylowl.proj.brightside.corpus import load_concrete_doc


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', type=str,
                        help='input directory path')
    parser.add_argument('output_dir', type=str,
                        help='output directory path')
    args = parser.parse_args()
    concrete_to_json(
        args.input_dir,
        args.output_dir,
    )


def concrete_to_json(input_dir, output_dir):
    input_output_paths = nested_input_output_file_paths(input_dir, output_dir,
                                                        path_is_concrete)
    for (input_path, output_path) in input_output_paths:
        doc = load_concrete_doc(input_path)
        output_path = os.path.splitext(output_path)[0] + '.json'
        with codecs.open(output_path, mode='w', encoding='utf-8') as f:
            d = dict()
            d['id'] = doc.id
            d['tokens'] = doc.tokens
            d['text'] = doc.text
            d['attrs'] = doc.attrs
            json.dump(d, f, indent=2)


if __name__ == '__main__':
    main()
