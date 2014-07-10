#!/usr/bin/env python


import os
import json
from pylowl.proj.brightside.corpus import load_vocab
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
    generate_d3_topic_graph(
        args.trunc_csv,
        args.vocab_path,
        lambda_ss_filename=args.lambda_ss,
        Elogpi_filename=args.Elogpi,
        logEpi_filename=args.logEpi,
        Elogtheta_filename=args.Elogtheta,
        logEtheta_filename=args.logEtheta,
        output_filename=args.output_path
    )


def generate_d3_topic_graph(trunc_csv,
        vocab_filename,
        lambda_ss_filename,
        Elogpi_filename,
        logEpi_filename,
        Elogtheta_filename,
        logEtheta_filename,
        output_filename):

    vocab = load_vocab(vocab_filename)

    node_topics = []
    graph = {}

    trunc = [int(t) for t in trunc_csv.split(',')]
    m = tree_index_m(trunc)
    b = tree_index_b(trunc)

    for node in tree_iter(trunc):
        node_topics.append({
            'children': [],
            'words': [{'word': vocab[t]} for t in range(len(vocab))]
        })

    for (stat_name, stat_filename) in (('Elogtheta', Elogtheta_filename),
                                       ('logEtheta', logEtheta_filename),
                                       ('lambda_ss', lambda_ss_filename)):
        with open(stat_filename) as f:
            for (idx, line) in enumerate(f):
                for (t, w) in enumerate(float(w) for w in line.strip().split()):
                    node_topics[idx]['words'][t][stat_name] = w

    # TODO remove this later
    from pylowl.proj.brightside.utils import take
    def copy_k(d, k, new_k):
        new_d = d.copy()
        new_d[new_k] = d[k]
        return new_d
    node_topics = [dict(children=[], words=list(copy_k(d, 'lambda_ss', 'weight') for d in take(sorted(node_topic['words'], key=lambda p: p['lambda_ss'], reverse=True), 10)))
                   for node_topic in node_topics]

    for (stat_name, stat_filename) in (('Elogpi', Elogpi_filename),
                                       ('logEpi', logEpi_filename)):
        with open(stat_filename) as f:
            for (idx, line) in enumerate(f):
                w = float(line.strip())
                node_topics[idx][stat_name] = w

    for node in tree_iter(trunc):
        idx = tree_index(node, m, b)
        node_dict = node_topics[idx]
        graph[node] = node_dict
        parent = node[:-1]
        if parent:
            graph[parent]['children'].append(node_dict)

    with open(output_filename, 'w') as f:
        json.dump(graph[(0,)], f, indent=2)


if __name__ == '__main__':
    main()
