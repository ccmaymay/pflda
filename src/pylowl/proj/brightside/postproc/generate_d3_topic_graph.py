#!/usr/bin/env python


import os
import json
from pylowl.proj.brightside.utils import take, load_vocab, tree_index_m, tree_index_b, tree_iter, tree_index


DEFAULT_NUM_WORDS_PER_TOPIC = 10


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(
        lambda_ss=None,
        Elogpi=None,
        logEpi=None,
        Elogtheta=None,
        logEtheta=None,
        num_words_per_topic=DEFAULT_NUM_WORDS_PER_TOPIC,
    )
    parser.add_argument('trunc_csv', type=str,
                        help='comma-separated list of truncations (per level)')
    parser.add_argument('vocab_path', type=str,
                        help='vocab file path')
    parser.add_argument('output_path', type=str,
                        help='output file path')
    parser.add_argument('--lambda_ss', type=str,
                        help='lambda_ss file path')
    parser.add_argument('--Elogpi', type=str,
                        help='Elogpi file path')
    parser.add_argument('--logEpi', type=str,
                        help='logEpi file path')
    parser.add_argument('--Elogtheta', type=str,
                        help='Elogtheta file path')
    parser.add_argument('--logEtheta', type=str,
                        help='logEtheta file path')
    parser.add_argument('--num_words_per_topic', type=int,
                        help='number of words to write per topic')

    args = parser.parse_args()
    generate_d3_topic_graph(
        args.trunc_csv,
        args.vocab_path,
        args.output_path,
        lambda_ss_filename=args.lambda_ss,
        Elogpi_filename=args.Elogpi,
        logEpi_filename=args.logEpi,
        Elogtheta_filename=args.Elogtheta,
        logEtheta_filename=args.logEtheta,
        num_words_per_topic=args.num_words_per_topic
    )


def sorted_topic_word_weight_lists(input_filename, vocab):
    with open(input_filename) as f:
        for line in f:
            yield sorted(
                ((vocab[t], w) for (t, w) in
                    enumerate([float(w) for w in line.strip().split()])),
                key=lambda p: p[1],
                reverse=True)


def generate_d3_topic_graph(trunc_csv, vocab_filename, output_filename,
        lambda_ss_filename=None,
        Elogpi_filename=None,
        logEpi_filename=None,
        Elogtheta_filename=None,
        logEtheta_filename=None,
        num_words_per_topic=DEFAULT_NUM_WORDS_PER_TOPIC):

    vocab = load_vocab(vocab_filename)

    node_topics = []
    graph = {}

    # TODO how to select words/sort?!
    for ww_list in sorted_topic_word_weight_lists(input_filename, vocab):
        words = []
        for (word, weight) in take(ww_list, num_words_per_topic):
            words.append({
                'word': word,
                'weight': weight
            })
        node_topics.append({
            'words': words,
            'weight': 0 # TODO
            'children': []
        })

    trunc = [int(t) for t in trunc_csv.split(',')]
    m = tree_index_m(trunc)
    b = tree_index_b(trunc)
    for node in tree_iter(trunc):
        idx = tree_index(node, m, b)
        parent = node[:-1]
        graph[node] = node_topics[idx]
        if parent:
            graph[parent]['children'].append(node_dict)

    with open(output_filename, 'w') as f:
        json.dump(graph[(0,)], f, indent=2)


if __name__ == '__main__':
    main()
