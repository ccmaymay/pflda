#!/usr/bin/env python


import os
from utils import nested_file_paths
from pylowl.proj.brightside.corpus import update_concrete_comms


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', type=str,
                        help='data directory path')

    args = parser.parse_args()
    tng_set_user_to_category(args.data_dir)


def set_user(comm):
    category = os.path.basename(os.path.dirname(self.id))
    comm.attrs['user'] = category


def tng_set_user_to_category(data_dir):
    update_concrete_comms(nested_file_paths(data_dir))


if __name__ == '__main__':
    main()
