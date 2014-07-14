#!/usr/bin/env python


from pylowl.proj.brightside.corpus import load_concrete_docs
from pylowl.proj.brightside.utils import nested_file_paths


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', type=str,
                        help='input directory path')

    args = parser.parse_args()
    count_users(nested_file_paths(args.input_dir))


def count_users(paths):
    users = dict()
    for doc in load_concrete_docs(paths):
        user = doc.attrs['user']
        if user in users:
            users[user] += 1
        else:
            users[user] = 1

    users_items = users.items()
    users_items.sort(key=lambda p: p[1], reverse=True)
    for (k, v) in users_items:
        print '%12d %s' % (v, k)
    print

    print 'Number of users: %d' % len(users)


if __name__ == '__main__':
    main()
