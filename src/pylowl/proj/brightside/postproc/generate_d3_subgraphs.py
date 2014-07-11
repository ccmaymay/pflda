#!/usr/bin/env python


import re
import json
import itertools as it
from pylowl.proj.brightside.corpus import load_vocab, load_concrete_raw
from pylowl.proj.brightside.utils import tree_index_m, tree_index_b, tree_iter, tree_index
import datetime


EPOCH = datetime.datetime(1970, 1, 1)
DATETIME_FORMAT = ''


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('trunc_csv', type=str,
                        help='comma-separated list of truncations (per level)')
    parser.add_argument('vocab_path', type=str,
                        help='vocab file path')
    parser.add_argument('data_path', type=str,
                        help='data path')
    parser.add_argument('output_path', type=str,
                        help='output file path')
    parser.add_argument('--subtree', type=str, required=True,
                        help='subtree file path')
    parser.add_argument('--lambda_ss', type=str, required=True,
                        help='lambda_ss file path')
    parser.add_argument('--doc_lambda_ss', type=str, required=True,
                        help='per-doc lambda_ss file path')
    parser.add_argument('--Elogpi', type=str, required=True,
                        help='Elogpi file path')
    parser.add_argument('--logEpi', type=str, required=True,
                        help='logEpi file path')

    args = parser.parse_args()

    generate_d3_subgraphs(
        args.trunc_csv,
        args.vocab_path,
        args.data_path,
        subtree_filename=args.subtree,
        lambda_ss_filename=args.lambda_ss,
        doc_lambda_ss_filename=args.doc_lambda_ss,
        Elogpi_filename=args.Elogpi,
        logEpi_filename=args.logEpi,
        output_filename=args.output_path,
    )


def expectation(p, x):
    if x:
        return sum([p_i*x_i for (p_i, x_i) in zip(p, x)])
    else:
        return None


def parse_datetime(datetime_str):
    return datetime.strptime(datetime_str, DATETIME_FORMAT)


def datetime_to_float(d):
    return (d - EPOCH).total_seconds()


def generate_d3_subgraphs(trunc_csv,
        vocab_filename,
        data_path,
        subtree_filename,
        lambda_ss_filename,
        doc_lambda_ss_filename,
        Elogpi_filename,
        logEpi_filename,
        output_filename):

    vocab = load_vocab(vocab_filename)

    trunc = [int(t) for t in trunc_csv.split(',')]
    m = tree_index_m(trunc)
    b = tree_index_b(trunc)

    subtree_dicts_per_user = {}
    node_maps_per_user = {}

    with open(subtree_filename) as subtree_f, \
         open(Elogpi_filename) as Elogpi_f, \
         open(logEpi_filename) as logEpi_f, \
         open(lambda_ss_filename) as lambda_ss_f:
        for (subtree_line, stat_lines) in it.izip(subtree_f, it.izip(
                Elogpi_f, logEpi_f, lambda_ss_f)):
            pieces = subtree_line.strip().split()
            user = pieces[0]
            node_map = dict((p[1], p[0]) for p in
                            enumerate([int(i) for i in pieces[1:]]))

            subtree_dicts = [None] * len(list(tree_iter(trunc)))
            for node in tree_iter(trunc):
                idx = tree_index(node, m, b)
                active = idx in node_map
                subtree_dicts[idx] = {
                    'active': active,
                    'children': [],
                }

            for (stat_name, stat_line) in it.izip(
                    ('Elogpi', 'logEpi', 'lambda_ss'),
                    stat_lines):
                stat_pieces = stat_line.strip().split()
                if user != stat_pieces[0]:
                    raise Exception('user do not match: %s and %s'
                                    % (user, stat_pieces[0]))
                weights = [float(w) for w in stat_pieces[1:]]
                for node in tree_iter(trunc):
                    idx = tree_index(node, m, b)
                    if idx in node_map:
                        subtree_dicts[idx][stat_name] = weights[node_map[idx]]

            subtree_dicts_per_user[user] = subtree_dicts
            node_maps_per_user[user] = node_map

    doc_id_path_map = dict()
    data_loc = glob(data_path)
    for doc in load_concrete(data_loc):
        doc_id_path_map[doc.attrs['identifier']] = doc.path

    with open(doc_lambda_ss_filename) as doc_lambda_ss_f:
        for line in doc_lambda_ss_f:
            pieces = line.strip().split()
            user = pieces[0]
            doc_identifier = pieces[1]
            doc = load_concrete(doc_id_path_map[doc_identifier])
            datetime_float = datetime_to_float(parse_datetime(doc.timestamp))
            latitude = doc.latitude
            longitude = doc.longitude

            weights = [float(w) for w in pieces[2:]]
            weights_sum = sum(weights)
            probabilities = [w/weights_sum for w in weights]

            node_map = node_maps_per_user[user]
            subtree_dicts = subtree_dicts_per_user[user]
            for node in tree_iter(trunc):
                idx = tree_index(node, m, b)
                if idx in node_map:
                    Edatetime = expectation(probabilities, datetime_floats)
                    subtree_dicts[idx]['expected_time'] = Edatetime

    json_data = []

    for (user, subtree_dicts) in subtree_dicts_per_user.items():
        num_active = 0
        for node in tree_iter(trunc):
            idx = tree_index(node, m, b)
            p = node[:-1]
            num_active += subtree_dicts[idx]['active']
            if p:
                p_idx = tree_index(p, m, b)
                subtree_dicts[p_idx]['children'].append(subtree_dicts[idx])
        json_data.append({
            'user': user,
            'subtree': subtree_dicts[0],
            'num_active': num_active
        })

    json_data.sort(key=lambda s: s['num_active'], reverse=True)

    with open(output_filename, 'w') as f:
        json.dump(json_data, f, indent=2)


if __name__ == '__main__':
    main()
