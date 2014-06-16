#!/usr/bin/env python


import sys
from pylowl.proj.brightside.utils import load_vocab


def _map_vocab(i, vocab):
    if ':' in i:
        pos = i.find(':')
        return vocab[int(i[:pos])] + i[pos:]
    else:
        return vocab[int(i)]


def main(vocab_filename):
    vocab = load_vocab(vocab_filename)
    for line in sys.stdin:
        print ' '.join(_map_vocab(i, vocab) for i in line.strip().split())


if __name__ == '__main__':
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
