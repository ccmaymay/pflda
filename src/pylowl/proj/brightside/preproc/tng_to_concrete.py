#!/usr/bin/env python


import re
from pylowl.proj.brightside.utils import nested_file_paths
from pylowl.proj.brightside.corpus import write_concrete_docs, Document


GREEDY_EMAIL_RE = re.compile(r'\S+@\S+\.\S+')
EMAIL_HISTORY_RE = re.compile(r'[^a-zA-Z0-9("\']')
WRITES_LINE_RE = re.compile(r'In article\b|.* (?:writes:|wrote:)$')
EMAIL_HEADER_RE = re.compile(r'(?:from|to|subject|date|reply-to|archive-name|version|last-modified):', re.IGNORECASE)

WHITESPACE_RE = re.compile(r'\s+')


def contains_email(token):
    return GREEDY_EMAIL_RE.match(token) is not None


def filter_tokens(tokens, remove_emails, remove_non_ascii):
    return [token for token in tokens
            if token
                and not (remove_emails and contains_email(token))
                and not (remove_non_ascii and sum(ord(c) > 127 for c in token) > 0)]


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', type=str,
                        help='input directory path')
    parser.add_argument('output_dir', type=str,
                        help='output directory path')
    parser.add_argument('--remove_first_paragraph', action='store_true',
                        help='remove first block of text in each document')
    parser.add_argument('--remove_email_headers', action='store_true',
                        help='remove email headers (From:, Subject:, etc.)')
    parser.add_argument('--remove_walls', action='store_true',
                        help='remove walls of text (lines without spaces)')
    parser.add_argument('--remove_emails', action='store_true',
                        help='remove email addresses')
    parser.add_argument('--remove_email_history', action='store_true',
                        help='remove lines starting with >, ), etc.')
    parser.add_argument('--remove_writes_lines', action='store_true',
                        help='remove "John Doe writes:" lines (from inline email responses)')
    parser.add_argument('--remove_non_ascii', action='store_true',
                        help='remove non-ascii tokens')

    args = parser.parse_args()
    tng_to_concrete(
        args.input_dir,
        args.output_dir,
        remove_first_paragraph=args.remove_first_paragraph,
        remove_email_headers=args.remove_email_headers,
        remove_walls=args.remove_walls,
        remove_emails=args.remove_emails,
        remove_email_history=args.remove_email_history,
        remove_writes_lines=args.remove_writes_lines,
        remove_non_ascii=args.remove_non_ascii,
    )


def tng_to_concrete(input_dir, output_dir, remove_first_paragraph=False,
         remove_email_headers=False, remove_walls=False, remove_emails=False,
         remove_email_history=False, remove_writes_lines=False,
         remove_non_ascii=False):

    def iter_docs():
        for input_path in nested_file_paths(input_dir):
            tokens = []
            with open(input_path) as f:
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
                            tokens.extend(filter_tokens(line.split(),
                                                        remove_emails,
                                                        remove_non_ascii))
                    else:
                        seen_empty_line = True

            text = ' '.join(tokens)
            yield Document(tokens, text=text, id=input_path)

    write_concrete_docs(iter_docs(), output_dir)


if __name__ == '__main__':
    main()
