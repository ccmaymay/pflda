#!/usr/bin/env python


import json
import itertools as it
from pylowl.proj.brightside.utils import tree_index_m, tree_index_b, tree_iter, tree_index


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

    args = parser.parse_args()
    generate_d3_subgraphs(
        args.trunc_csv,
        args.vocab_path,
        subtree_filename=args.subtree,
        lambda_ss_filename=args.lambda_ss,
        Elogpi_filename=args.Elogpi,
        logEpi_filename=args.logEpi,
        Elogtheta_filename=args.Elogtheta,
        logEtheta_filename=args.logEtheta,
        output_filename=args.output_path
    )


def generate_d3_subgraphs(trunc_csv,
        vocab_filename,
        subtree_filename,
        lambda_ss_filename,
        Elogpi_filename,
        logEpi_filename,
        Elogtheta_filename,
        logEtheta_filename,
        output_filename):

    vocab = load_vocab(vocab_filename)

    trunc = [int(t) for t in trunc_csv.split(',')]
    m = tree_index_m(trunc)
    b = tree_index_b(trunc)

    subtrees = {}
    subtree_nodes = [None] * len(list(tree_iter(trunc)))

    with open(subtree_filename) as f:
        for line in f:
            pieces = line.strip().split()
            identifier = pieces[0]
            node_ids = set([int(i) for i in pieces[1:]])
            for node in tree_iter(trunc):
                idx = tree_index(node, m, b)
                active = idx in node_ids
                subtree_nodes[idx] = {
                    'active': active,
                    'children': []
                }
                p = node[:-1]
                if p:
                    p_idx = tree_index(p, m, b)
                    subtree_nodes[p_idx]['children'].append(subtree_nodes[idx])
            subtrees[identifier] = {
                'identifier': identifier,
                'subtree': subtree_nodes[0]
            }

        f.seek(0)

        for (stat_name, stat_filename) in (('Elogtheta', Elogtheta_filename),
                                           ('logEtheta', logEtheta_filename),
                                           ('Elogpi', Elogpi_filename),
                                           ('logEpi', logEpi_filename),
                                           ('lambda_ss', lambda_ss_filename)):
            with open(stat_filename) as stat_f:
                for (line, stat_line) in it.izip(f, stat_f):
                    pieces = line.strip().split()
                    identifier = pieces[0]
                    node_ids = set([int(i) for i in pieces[1:]])

                    stat_pieces = stat_line.strip().split()
                    if identifier != stat_pieces[0]:
                        raise Exception('identifiers do not match: %s and %s'
                                        % (identifier, stat_pieces[0]))
                    weights = [float(w) for w in stat_pieces[1:]]

                    node_weights = dict(zip(node_ids, weights))
                    for node in tree_iter(trunc):
                        idx = tree_index(node, m, b)
                        if idx in node_weights:
                            subtrees[identifier]['subtree']

                for (idx, line) in enumerate(f):
                    for (t, w) in enumerate(float(w) for w in line.strip().split()):
                        node_topics[idx]['words'][t][stat_name] = w


                for node in tree_iter(trunc):
                    idx = tree_index(node, m, b)

            f.seek(0)
            node_weights.append({
                'children': [],
                'words': [{'word': vocab[t]} for t in range(len(vocab))]
            })

        json_data = []

        for (identifier, node_ids) in subtrees.items():
            d = dict(identifier=identifier)
            subtree_nodes = [None] * len(list(tree_iter(trunc)))

            for node in tree_iter(trunc):
                idx = tree_index(node, m, b)
                active = idx in node_ids
                subtree_nodes[idx] = dict(active=active, children=[])
                p = node[:-1]
                if p:
                    p_idx = tree_index(p, m, b)
                    subtree_nodes[p_idx]['children'].append(subtree_nodes[idx])

            d['subtree'] = subtree_nodes[0]
            json_data.append(d)

        with open(output_filename, 'w') as f:
            json.dump(json_data, f, indent=2)


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
