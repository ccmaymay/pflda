#!/bin/bash


from glob import glob
import re


def main(path_pattern, *line_patterns):
    line_res = [re.compile(line_pattern) for line_pattern in line_patterns]
    for path in glob(path_pattern):
        unmatched_res = dict(enumerate(line_res))
        with open(path) as f:
            for line in f:
                new_unmatched_res = None
                for (i, r) in unmatched_res.items():
                    if r.search(line) is not None:
                        if new_unmatched_res is None:
                            new_unmatched_res = unmatched_res.copy()
                        del new_unmatched_res[i]
                if new_unmatched_res is not None:
                    unmatched_res = new_unmatched_res
        if not unmatched_res:
            print path
                    

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
