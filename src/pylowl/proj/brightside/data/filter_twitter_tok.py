#!/usr/bin/env python


import sys
import re


ALPHA_RE = re.compile(r'[a-zA-Z]')


def is_bad(token, stop_set, idf, idf_threshold):
    return token.startswith('@') or ALPHA_RE.search(token) is None or token in stop_set or idf[token] < idf_threshold


def load_stop_set(filename):
    stop_set = set()
    with open(filename) as f:
        for line in f:
            stop_set.add(line.strip())
    return stop_set


def main(filename, stop_list=None, idf_threshold=None):
    if stop_list is None:
        stop_set = set()
    else:
        stop_set = load_stop_set(stop_list)

    if idf_threshold is None:
        idf_threshold = 0
    else:
        idf_threshold = float(idf_threshold)

    with open(filename) as f:
        df = dict()
        for line in f:
            doc_types = set()
            for token in line.strip().lower().split():
                doc_types.add(token)
            for t in doc_types:
                if t in df:
                    df[t] += 1
                else:
                    df[t] = 1
        idf = dict((t, 1.0/df) for (t, df) in df.items())

        items = idf.items()
        items.sort(key=lambda item: item[1])

        sys.stderr.write('idf:')
        for (k, v) in items:
            sys.stderr.write('%40s    %f\n' % (k, v))

        f.seek(0)

        for line in f:
            good_tokens = []
            for token in line.strip().lower().split():
                if not is_bad(token, stop_set, idf, idf_threshold):
                    good_tokens.append(token)
            if good_tokens:
                print ' '.join(good_tokens)


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
