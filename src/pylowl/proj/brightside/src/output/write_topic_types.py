#!/usr/bin/env python


import utils
import subprocess
import tempfile
import os


def main(vocab_filename, input_filename, output_filename):
    with open(vocab_filename) as f:
        num_types = 0
        for line in f:
            num_types += 1
        vocab = [None for t in xrange(num_types)]
        f.seek(0)
        for line in f:
            id_word_pair = line.strip().split()
            vocab[int(id_word_pair[0])] = id_word_pair[1]

    with open(output_filename, 'w') as out_f:
        with open(input_filename) as f:
            for line in f:
                weights = sorted(enumerate([float(w) for w in line.strip().split()]), key=lambda p: p[1], reverse=True)
                out_f.write(' '.join('%s %f' % (vocab[t], w) for (t, w) in weights) + '\n')


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
