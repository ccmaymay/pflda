#!/usr/bin/env python


def increment(d, k):
    if k in d:
        d[k] += 1
    else:
        d[k] = 1


def replace(token, vocab, rep_str=None):
    if vocab[token] == 1:
        return rep_str
    else:
        return token


def main(filename, rep_str=None):
    with open(filename) as f:
        vocab = dict()
        for line in f:
            for token in line.strip().split():
                increment(vocab, token)
        f.seek(0)
        for line in f:
            s = ' '.join((replace(token, vocab, rep_str) for token in line.strip().split() if replace(token, vocab, rep_str) is not None))
            if s:
                print s


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
