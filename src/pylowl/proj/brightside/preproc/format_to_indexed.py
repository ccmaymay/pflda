#!/usr/bin/env python


import logging
import os
import re
from pylowl.proj.brightside.preproc.utils import write_vocab, write_docs, make_parent_dir, load_word_set


NON_ALPHA_RE = re.compile(r'[^a-zA-Z]')


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(
        stop_list=None,
        dictionary=None,
        idf_lb=0.,
        idf_ub=1.,
        remove_non_alpha=False,
        lowercase=False,
        min_word_len=1,
        log_level='INFO'
    )
    parser.add_argument('train_input_filename', type=str,
                        help='doc-per-line input file path (training split)')
    parser.add_argument('test_input_filename', type=str,
                        help='doc-per-line input file path (testing split)')
    parser.add_argument('train_output_filename', type=str,
                        help='output file path (training split)')
    parser.add_argument('test_output_filename', type=str,
                        help='output file path (testing split)')
    parser.add_argument('vocab_output_filename', type=str,
                        help='vocab output file path')
    parser.add_argument('--stop_list', type=str,
                        help='path to stop list')
    parser.add_argument('--dictionary', type=str,
                        help='path to dictionary (whitelist)')
    parser.add_argument('--idf_lb', type=float,
                        help='inclusive IDF lower bound')
    parser.add_argument('--idf_ub', type=float,
                        help='inclusive IDF upper bound')
    parser.add_argument('--remove_non_alpha', action='store_true',
                        help='remove non-alphabet characters')
    parser.add_argument('--lowercase', action='store_true',
                        help='convert all tokens to lower case')
    parser.add_argument('--min_word_len', type=int,
                        help='minimum word length for inclusion')
    parser.add_argument('--log_level', type=str,
                        help='log level')

    args = parser.parse_args()
    format_to_indexed(
        args.train_input_filename,
        args.test_input_filename,
        args.train_output_filename,
        args.test_output_filename,
        args.vocab_output_filename,
        stop_list=args.stop_list,
        dictionary=args.dictionary,
        idf_lb=args.idf_lb,
        idf_ub=args.idf_ub,
        remove_non_alpha=args.remove_non_alpha,
        lowercase=args.lowercase,
        min_word_len=args.min_word_len,
        log_level=args.log_level
    )


def is_bad(token, stop_set, idf, idf_lb, idf_ub, dictionary_set, min_word_len):
    return (
        (dictionary_set is not None and token not in dictionary_set)
        or (stop_set is not None and token in stop_set)
        or idf[token] < idf_lb
        or idf[token] > idf_ub
        or len(token) < min_word_len
    )


def transform_token(token, remove_non_alpha, lowercase):
    if lowercase:
        token = token.lower()
    if remove_non_alpha:
        token = NON_ALPHA_RE.subn('', token)[0]
    return token


def format_to_indexed(train_input_filename, test_input_filename, train_output_filename, test_output_filename, vocab_output_filename, stop_list=None, dictionary=None, idf_lb=0., idf_ub=1., remove_non_alpha=False, lowercase=False, min_word_len=1, log_level='INFO'):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if dictionary is None:
        dictionary_set = None
    else:
        raw_dictionary_set = load_word_set(dictionary)
        dictionary_set = set()
        for word in raw_dictionary_set:
            if lowercase:
                word = word.lower()
            # hack for unix dict possessive madness
            if word.endswith("'s"):
                word = word[:-2]
            if word:
                dictionary_set.add(word)

    if stop_list is None:
        stop_set = None
    else:
        stop_set = load_word_set(stop_list)

    tt = lambda token: transform_token(token, remove_non_alpha, lowercase)

    vocab = dict()
    with open(train_input_filename) as f:
        df = dict()
        for line in f:
            doc_types = set()
            for token in line.strip().split():
                token = tt(token)
                doc_types.add(token)
            for t in doc_types:
                if t in df:
                    df[t] += 1
                else:
                    df[t] = 1
        idf = dict((t, 1.0/df) for (t, df) in df.items())

        items = idf.items()
        items.sort(key=lambda item: item[1])

        logging.info('idf:')
        for (k, v) in items:
            logging.info('%09f %s' % (v, k))

        f.seek(0)

        for line in f:
            for token in line.strip().split():
                token = transform_token(token, remove_non_alpha, lowercase)
                if not is_bad(token, stop_set, idf, idf_lb, idf_ub, dictionary_set, min_word_len):
                    if token not in vocab:
                        vocab[token] = len(vocab)

    write_vocab(vocab_output_filename, vocab)

    write_docs(train_input_filename, train_output_filename, vocab, tt)
    write_docs(test_input_filename, test_output_filename, vocab, tt)


if __name__ == '__main__':
    main()
