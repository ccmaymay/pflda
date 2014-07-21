#!/usr/bin/env python


import os
from pylowl.proj.brightside.utils import nested_file_paths
from pylowl.proj.brightside.corpus import update_concrete_comms


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', type=str,
                        help='data directory path')
    parser.add_argument('attr', type=str,
                        help='name of attribute to set')

    args = parser.parse_args()
    tng_set_attr_to_category(args.data_dir, args.attr)


def set_attr(comm, attr):
    category = os.path.basename(os.path.dirname(comm.id))
    comm.keyValueMap[attr] = category


def tng_set_attr_to_category(data_dir, attr):
    update_concrete_comms(nested_file_paths(data_dir),
                          lambda c: set_attr(c, attr))


if __name__ == '__main__':
    main()
