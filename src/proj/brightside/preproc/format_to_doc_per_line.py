#!/usr/bin/env python


import os
from utils import make_parent_dir


def main(input_path, output_path):
    if not os.path.isdir(input_path):
        raise Exception('"%s" does not seem to be a directory' % input_path)
    make_parent_dir(output_path)
    with open(output_path, 'w') as out_f:
        for (dir_path, dir_entries, file_entries) in os.walk(input_path):
            for filename in file_entries:
                input_file_path = os.path.join(dir_path, filename)
                with open(input_file_path) as f:
                    i = 0
                    for line in f:
                        if i > 0:
                            out_f.write(' ')
                        out_f.write(line.strip())
                        i += 1
                    out_f.write('\n')


if __name__ == '__main__':
    import sys

    args = []
    params = dict()
    for token in sys.argv[1:]:
        eq_pos = token.find('=')
        if token.startswith('--') and eq_pos >= 0:
            k = token[len('--'):eq_pos]
            v = token[(eq_pos+1):len(token)]
            params[k] = v
        else:
            args.append(token)

    main(*args, **params)
