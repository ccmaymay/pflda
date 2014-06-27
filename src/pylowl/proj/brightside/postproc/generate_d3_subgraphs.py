#!/usr/bin/env python


import re
import json
import itertools as it
from pylowl.proj.brightside.utils import load_vocab, tree_index_m, tree_index_b, tree_iter, tree_index


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('trunc_csv', type=str,
                        help='comma-separated list of truncations (per level)')
    parser.add_argument('vocab_path', type=str,
                        help='vocab file path')
    parser.add_argument('output_path', type=str,
                        help='output file path')
    parser.add_argument('--subtree', type=str, required=True,
                        help='subtree file path')
    parser.add_argument('--lambda_ss', type=str, required=True,
                        help='lambda_ss file path')
    parser.add_argument('--Elogpi', type=str, required=True,
                        help='Elogpi file path')
    parser.add_argument('--logEpi', type=str, required=True,
                        help='logEpi file path')
    parser.add_argument('--Elogtheta', type=str, required=True,
                        help='Elogtheta file path')
    parser.add_argument('--logEtheta', type=str, required=True,
                        help='logEtheta file path')
    parser.add_argument('--identifier_re', type=str,
                        help='regex to filter document identifiers')

    args = parser.parse_args()
    if args.identifier_re is None:
        identifier_re = None
    else:
        identifier_re = re.compile(args.identifier_re)

    generate_d3_subgraphs(
        args.trunc_csv,
        args.vocab_path,
        subtree_filename=args.subtree,
        lambda_ss_filename=args.lambda_ss,
        Elogpi_filename=args.Elogpi,
        logEpi_filename=args.logEpi,
        Elogtheta_filename=args.Elogtheta,
        logEtheta_filename=args.logEtheta,
        output_filename=args.output_path,
        identifier_re=identifier_re
    )


def generate_d3_subgraphs(trunc_csv,
        vocab_filename,
        subtree_filename,
        lambda_ss_filename,
        Elogpi_filename,
        logEpi_filename,
        Elogtheta_filename,
        logEtheta_filename,
        output_filename,
        identifier_re=None):

    vocab = load_vocab(vocab_filename)

    trunc = [int(t) for t in trunc_csv.split(',')]
    m = tree_index_m(trunc)
    b = tree_index_b(trunc)

    subtree_dicts_per_id = {}

    with open(subtree_filename) as subtree_f, \
         open(Elogtheta_filename) as Elogtheta_f, \
         open(logEtheta_filename) as logEtheta_f, \
         open(Elogpi_filename) as Elogpi_f, \
         open(logEpi_filename) as logEpi_f, \
         open(lambda_ss_filename) as lambda_ss_f:
        for (subtree_line, stat_lines) in it.izip(subtree_f, it.izip(
                Elogtheta_f, logEtheta_f, Elogpi_f, logEpi_f, lambda_ss_f)):
            pieces = subtree_line.strip().split()
            identifier = pieces[0]
            if identifier_re is not None and identifier_re.match(identifier) is None:
                continue
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
                    ('Elogtheta', 'logEtheta', 'Elogpi', 'logEpi', 'lambda_ss'),
                    stat_lines):
                stat_pieces = stat_line.strip().split()
                if identifier != stat_pieces[0]:
                    raise Exception('identifiers do not match: %s and %s'
                                    % (identifier, stat_pieces[0]))
                weights = [float(w) for w in stat_pieces[1:]]
                for node in tree_iter(trunc):
                    idx = tree_index(node, m, b)
                    if idx in node_map:
                        subtree_dicts[idx][stat_name] = weights[node_map[idx]]

            subtree_dicts_per_id[identifier] = subtree_dicts

    json_data = []

    for (identifier, subtree_dicts) in subtree_dicts_per_id.items():
        num_active = 0
        for node in tree_iter(trunc):
            idx = tree_index(node, m, b)
            p = node[:-1]
            num_active += subtree_dicts[idx]['active']
            if p:
                p_idx = tree_index(p, m, b)
                subtree_dicts[p_idx]['children'].append(subtree_dicts[idx])
        json_data.append({
            'identifier': identifier,
            'subtree': subtree_dicts[0],
            'num_active': num_active
        })

    json_data.sort(key=lambda s: s['num_active'], reverse=True)

    with open(output_filename, 'w') as f:
        json.dump(json_data, f, indent=2)


if __name__ == '__main__':
    main()
