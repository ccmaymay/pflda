#!/usr/bin/env python


from glob import glob
import re
import itertools as it
from pylowl.proj.brightside.corpus import Document, write_concrete


SPLIT_RE = re.compile(r'\W+')


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def docs_to_concrete(input_pattern, output_dir):
    paths = glob(input_pattern)
    paths.sort()
    docs = []
    for path in paths:
        tokens = []
        with open(path) as f:
            for line in f:
                tokens.extend(token for token in SPLIT_RE.split(line)
                              if token and not is_num(token))
        if tokens:
            docs.append(Document(tokens, id=path))
    write_concrete(docs, output_dir)


if __name__ == '__main__':
    import sys
    docs_to_concrete(*sys.argv[1:])
