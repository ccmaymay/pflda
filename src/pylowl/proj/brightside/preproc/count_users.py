#!/usr/bin/env python


from glob import glob
from pylowl.proj.brightside.corpus import load_concrete


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', type=str,
                        help='input directory path')

    args = parser.parse_args()
    count_users(args.input_path)


def count_users(input_path):
    input_loc = glob(input_path)

    users = dict()
    for doc in load_concrete(input_loc):
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
