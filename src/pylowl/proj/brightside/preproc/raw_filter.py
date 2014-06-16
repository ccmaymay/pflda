#!/usr/bin/env python


import os
import re
import gzip
from preproc.utils import get_path_suffix, input_output_paths
from utils import make_parent_dir


GREEDY_EMAIL_RE = re.compile(r'\S+@\S+\.\S+')
EMAIL_HISTORY_RE = re.compile(r'[^a-zA-Z0-9("\']')
WRITES_LINE_RE = re.compile(r'In article\b|.* (?:writes:|wrote:)$')
EMAIL_HEADER_RE = re.compile(r'(?:from|to|subject|date|reply-to|archive-name|version|last-modified):', re.IGNORECASE)

WHITESPACE_RE = re.compile(r'\s+')


def contains_email(token):
    return GREEDY_EMAIL_RE.match(token) is not None


def filter_tokens(tokens, remove_emails):
    return (token for token in tokens
            if token and not (remove_emails and contains_email(token)))


def main(input_path, output_path, remove_first_paragraph=False,
         remove_email_headers=False, remove_walls=False, remove_emails=False,
         remove_email_history=False, remove_writes_lines=False):
    for (input_file_path, output_file_path) in input_output_paths(input_path,
                                                                  output_path):
        with open(output_file_path, 'w') as out_f:
            with open(input_file_path) as f:
                seen_empty_line = False
                for line in f:
                    line = line.strip()
                    if line:
                        contains_ws = WHITESPACE_RE.search(line) is not None
                        email_history = EMAIL_HISTORY_RE.match(line) is not None
                        writes_line = WRITES_LINE_RE.match(line) is not None
                        email_header = EMAIL_HEADER_RE.match(line) is not None
                        if ((seen_empty_line or not remove_first_paragraph)
                            and (contains_ws or not remove_walls)
                            and not (remove_email_headers and email_header)
                            and not (remove_email_history and email_history)
                            and not (remove_writes_lines and writes_line)):

                            tokens = filter_tokens(line.split(), remove_emails)
                            out_f.write(' '.join(tokens) + '\n')
                    else:
                        seen_empty_line = True


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
