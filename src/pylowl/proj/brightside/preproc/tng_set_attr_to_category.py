#!/usr/bin/env python


import os
from utils import nested_file_paths
from pylowl.proj.brightside.corpus import update_concrete_comms


DEFAULT_ATTR = 'user'


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(attr=DEFAULT_ATTR)
    parser.add_argument('data_dir', type=str,
                        help='data directory path')
    parser.add_argument('--attr', type=str,
                        help='name of attribute to set')

    args = parser.parse_args()
    tng_set_user_to_category(args.data_dir, attr=args.attr)


def set_user(comm, attr):
    category = os.path.basename(os.path.dirname(self.id))
    comm.attrs[attr] = category


def tng_set_user_to_category(data_dir, attr=DEFAULT_ATTR):
    update_concrete_comms(nested_file_paths(data_dir),
                          lambda c: set_user(c, attr))


if __name__ == '__main__':
    main()
