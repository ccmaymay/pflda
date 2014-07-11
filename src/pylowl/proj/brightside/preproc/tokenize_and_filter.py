#!/usr/bin/env python


import os
import re
from glob import glob
from pylowl.proj.brightside.preproc.utils import load_word_set, write_vocab
from pylowl.proj.brightside.corpus import load_concrete, write_concrete, Document


DEFAULT_SPLIT_PATTERN = r'\s+'
IDF_WORD_PRINT_LIMIT = 100


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(
        stop_list=None,
        dictionary=None,
        split_pattern=DEFAULT_SPLIT_PATTERN,
        idf_lb=0.,
        idf_ub=1.,
        lowercase=False,
    )
    parser.add_argument('train_input_path', type=str,
                        help='input directory path (training split)')
    parser.add_argument('test_input_path', type=str,
                        help='input directory path (testing split)')
    parser.add_argument('train_output_path', type=str,
                        help='output directory path (training split)')
    parser.add_argument('test_output_path', type=str,
                        help='output directory path (testing split)')
    parser.add_argument('vocab_output_path', type=str,
                        help='vocab output file path')
    parser.add_argument('--split_pattern', type=str,
                        help='regex on which to split (tokenize)')
    parser.add_argument('--token_filter_pattern', type=str, action='append',
                        dest='token_filter_patterns',
                        help='regex for token that should be filtered out')
    parser.add_argument('--char_filter_pattern', type=str, action='append',
                        dest='char_filter_patterns',
                        help='regex for char that should be filtered out (within token)')
    parser.add_argument('--stop_list', type=str,
                        help='path to stop list')
    parser.add_argument('--dictionary', type=str,
                        help='path to dictionary (whitelist)')
    parser.add_argument('--idf_lb', type=float,
                        help='inclusive IDF lower bound')
    parser.add_argument('--idf_ub', type=float,
                        help='inclusive IDF upper bound')
    parser.add_argument('--lowercase', action='store_true',
                        help='convert all tokens to lower case')

    args = parser.parse_args()
    tokenize_and_filter(
        args.train_input_path,
        args.test_input_path,
        args.train_output_path,
        args.test_output_path,
        args.vocab_output_path,
        stop_list=args.stop_list,
        split_pattern=args.split_pattern,
        token_filter_patterns=args.token_filter_patterns,
        char_filter_patterns=args.char_filter_patterns,
        dictionary=args.dictionary,
        idf_lb=args.idf_lb,
        idf_ub=args.idf_ub,
        lowercase=args.lowercase,
    )


def filter_token(token, stop_set, idf, idf_lb, idf_ub, dictionary_set, *token_filter_res):
    return (
        (not token)
        or (dictionary_set is not None and token not in dictionary_set)
        or (stop_set is not None and token in stop_set)
        or idf[token] < idf_lb
        or idf[token] > idf_ub
        or sum([r.match(token) is not None for r in token_filter_res]) > 0
    )


def transform_token(token, lowercase, *char_filter_res):
    if lowercase:
        token = token.lower()
    for r in char_filter_res:
        token = r.subn('', token)[0]
    return token


def iter_docs(input_loc, tt, vocab, split_re):
    for doc in load_concrete(input_loc):
        tokens = [tt(token) for token in split_re.split(doc.text)]
        tokens = [token for token in tokens if token in vocab]
        if tokens:
            yield Document(tokens, text=doc.text, **doc.attrs)


def tokenize_and_filter(train_input_path, test_input_path,
                  train_output_path, test_output_path, vocab_output_path,
                  stop_list=None, dictionary=None, idf_lb=0., idf_ub=1.,
                  lowercase=False, split_pattern=DEFAULT_SPLIT_PATTERN,
                  char_filter_patterns=None, token_filter_patterns=None):
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

    if token_filter_patterns is None:
        token_filter_patterns = []

    if char_filter_patterns is None:
        char_filter_patterns = []

    split_re = re.compile(split_pattern)
    token_filter_res = [re.compile(r) for r in token_filter_patterns]
    char_filter_res = [re.compile(r) for r in char_filter_patterns]

    tt = lambda token: transform_token(token, lowercase, *char_filter_res)

    train_input_loc = glob(train_input_path)
    test_input_loc = glob(test_input_path)

    df = dict()
    for doc in load_concrete(train_input_loc):
        doc_types = set()
        for token in split_re.split(doc.text):
            token = tt(token)
            doc_types.add(token)
        for t in doc_types:
            if t in df:
                df[t] += 1
            else:
                df[t] = 1

    idf = dict((t, 1.0/df) for (t, df) in df.items())

    idf_items = idf.items()
    idf_items.sort(key=lambda item: item[1])

    print 'Low idf:'
    idx = 0
    for (k, v) in idf_items:
        if v >= idf_lb:
            break
        idx += 1
    for (k, v) in idf_items[max(0,idx-IDF_WORD_PRINT_LIMIT):idx+IDF_WORD_PRINT_LIMIT]:
        if v < idf_lb:
            print '(-) %06f %s' % (v, k)
        else:
            print '(+) %06f %s' % (v, k)
    print

    print 'High idf:'
    idx = 0
    for (k, v) in idf_items:
        if v > idf_ub:
            break
        idx += 1
    for (k, v) in idf_items[max(0,idx-IDF_WORD_PRINT_LIMIT):idx+IDF_WORD_PRINT_LIMIT]:
        if v <= idf_ub:
            print '(+) %06f %s' % (v, k)
        else:
            print '(-) %06f %s' % (v, k)
    print

    vocab = dict()
    for doc in load_concrete(train_input_loc):
        doc_types = set()
        for token in split_re.split(doc.text):
            token = tt(token)
            if not filter_token(token, stop_set, idf, idf_lb, idf_ub, dictionary_set, *token_filter_res):
                if token not in vocab:
                    vocab[token] = len(vocab)

    write_concrete(iter_docs(train_input_loc, tt, vocab, split_re), train_output_path)
    write_concrete(iter_docs(test_input_loc, tt, vocab, split_re), test_output_path)
    write_vocab(vocab_output_path, vocab)


if __name__ == '__main__':
    main()
