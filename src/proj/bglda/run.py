#!/usr/bin/env python


if __name__ == '__main__':
    import bglda
    import sys

    dataset_path = sys.argv[1]

    params = dict()
    for token in sys.argv[2:]:
        eq_pos = token.find('=')
        if token.startswith('--') and eq_pos >= 0:
            k = token[len('--'):eq_pos]
            v = token[(eq_pos+1):len(token)]
            params[k] = v

    bglda.run(dataset_path, **params)
