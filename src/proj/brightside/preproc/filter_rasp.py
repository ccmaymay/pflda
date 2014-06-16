#!/usr/bin/env python


import logging
import os
import re
import subprocess
import tempfile
from preproc.utils import input_output_paths


TOKEN_RE = re.compile(r'(\S+?)(?:\+\S*)?_[^_]+')
SENTENCE_START = r'^_^'
WRAP_RASP_ARGS = ('bash', 'wrap_rasp.sh')


def main(rasp_path, input_path, output_path):
    (temp_file_fd, temp_file_path) = tempfile.mkstemp()
    os.close(temp_file_fd)

    try:
        for (input_file_path, output_file_path) in input_output_paths(input_path, output_path):
            subprocess.call(WRAP_RASP_ARGS
                            + (rasp_path, input_file_path, temp_file_path))
            with open(temp_file_path) as f:
                with open(output_file_path, 'w') as out_f:
                    for line in f:
                        tokens = line.strip().split()
                        if tokens:
                            if tokens[0] == SENTENCE_START:
                                for (i, token) in enumerate(tokens[1:]):
                                    m = TOKEN_RE.match(token)
                                    if m is not None:
                                        if i > 0:
                                            out_f.write(' ')
                                        out_f.write(m.group(1))
                                    else:
                                        logging.warn('token error: %s\n' % token)
                                out_f.write('\n')
                            else:
                                logging.warn('sentence error: %s\n' % line)
    finally:
        os.remove(temp_file_path)


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
