#!/usr/bin/env python


import logging
import os
import re
from utils import write_vocab, write_data
from utils import load_word_set, make_parent_dir


NON_ALPHA_RE = re.compile(r'[^a-zA-Z]')


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


def main(train_input_filename, test_input_filename, train_output_filename, test_output_filename, vocab_output_filename, stop_list=None, dictionary=None, idf_lb=None, idf_ub=None, remove_non_alpha=False, lowercase=False, min_word_len=1, log_level=None):
    if log_level is None:
        log_level = 'INFO'
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if isinstance(min_word_len, str):
        min_word_len = int(min_word_len)

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

    if idf_lb is None:
        idf_lb = 0
    else:
        idf_lb = float(idf_lb)

    if idf_ub is None:
        idf_ub = 1
    else:
        idf_ub = float(idf_ub)

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
