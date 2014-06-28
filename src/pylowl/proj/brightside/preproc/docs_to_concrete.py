#!/usr/bin/env python


from glob import glob
import re
import itertools as it
from pylowl.proj.brightside.utils import write_concrete


SPLIT_RE = re.compile(r'\W+')


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def main(input_pattern, output_dir):
    paths = glob(input_pattern)
    paths.sort()
    docs = []
    for path in paths:
        doc = []
        with open(path) as f:
            for line in f:
                doc.extend(token for token in SPLIT_RE.split(line)
                           if token and not is_num(token))
        if doc:
            docs.append(doc)
    write_concrete(it.izip(docs, it.cycle([None])), output_dir)


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
