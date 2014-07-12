#!/usr/bin/env python


import re
import json
import itertools as it
from pylowl.proj.brightside.corpus import load_vocab
from pylowl.proj.brightside.utils import tree_index_m, tree_index_b, tree_iter, tree_index


DEFAULT_WORDS_PER_TOPIC=10


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(words_per_topic=DEFAULT_WORDS_PER_TOPIC)
    parser.add_argument('trunc_csv', type=str,
                        help='comma-separated list of truncations (per level)')
    parser.add_argument('vocab_path', type=str,
                        help='vocab file path')
    parser.add_argument('output_path', type=str,
                        help='output file path')
    parser.add_argument('--subtree', type=str, required=True,
                        help='subtree file path')
    parser.add_argument('--graph_lambda_ss', type=str, required=True,
                        help='graph_lambda_ss file path')
    parser.add_argument('--lambda_ss', type=str, required=True,
                        help='lambda_ss file path')
    parser.add_argument('--Elogpi', type=str, required=True,
                        help='Elogpi file path')
    parser.add_argument('--logEpi', type=str, required=True,
                        help='logEpi file path')
    parser.add_argument('--words_per_topic', type=int,
                        help='number of words to output per topic')

    args = parser.parse_args()

    generate_d3_subgraphs(
        args.trunc_csv,
        args.vocab_path,
        subtree_filename=args.subtree,
        graph_lambda_ss_filename=args.graph_lambda_ss,
        lambda_ss_filename=args.lambda_ss,
        Elogpi_filename=args.Elogpi,
        logEpi_filename=args.logEpi,
        output_filename=args.output_path,
        words_per_topic=args.words_per_topic,
    )


def generate_d3_subgraphs(trunc_csv,
        vocab_filename,
        subtree_filename,
        graph_lambda_ss_filename,
        lambda_ss_filename,
        Elogpi_filename,
        logEpi_filename,
        output_filename,
        words_per_topic=DEFAULT_WORDS_PER_TOPIC):

    vocab = load_vocab(vocab_filename)

    trunc = [int(t) for t in trunc_csv.split(',')]
    m = tree_index_m(trunc)
    b = tree_index_b(trunc)

    node_topics = []
    for node in tree_iter(trunc):
        node_topics.append({
            'words': [{'word': vocab[t]} for t in range(len(vocab))]
        })
    with open(graph_lambda_ss_filename) as f:
        for (idx, line) in enumerate(f):
            for (t, w) in enumerate(float(w) for w in line.strip().split()):
                node_topics[idx]['words'][t]['lambda_ss'] = w
    for node_dict in node_topics:
        node_dict['lambda_ss_sum'] = sum(d['lambda_ss'] for d in node_dict['words'])
        node_dict['words'].sort(key=lambda d: d['lambda_ss'], reverse=True)
        node_dict['words'] = node_dict['words'][:words_per_topic]

    for (stat_name, stat_filename) in (('Elogpi', Elogpi_filename),
                                       ('logEpi', logEpi_filename)):
        with open(stat_filename) as f:
            for (idx, line) in enumerate(f):
                w = float(line.strip())
                node_topics[idx][stat_name] = w
    subtree_dicts_per_id = {}

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
                node_dict = node_topics[idx]
                subtree_dicts[idx] = {
                    'active': active,
                    'children': [],
                    'words': node_dict['words'],
                    'global_lambda_ss_sum': node_dict['lambda_ss_sum'],
                }

            for (stat_name, stat_line) in it.izip(
                    ('Elogpi', 'logEpi', 'lambda_ss'),
                    stat_lines):
                stat_pieces = stat_line.strip().split()
                if user != stat_pieces[0]:
                    raise Exception('users do not match: %s and %s'
                                    % (user, stat_pieces[0]))
                weights = [float(w) for w in stat_pieces[1:]]
                for node in tree_iter(trunc):
                    idx = tree_index(node, m, b)
                    if idx in node_map:
                        subtree_dicts[idx][stat_name] = weights[node_map[idx]]

            subtree_dicts_per_id[user] = subtree_dicts

    json_data = []

    for (user, subtree_dicts) in subtree_dicts_per_id.items():
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
