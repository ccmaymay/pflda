#!/usr/bin/env python


import utils
import os
import json


NUM_TYPES_PER_TOPIC = 10


def sorted_topic_word_weight_lists(input_filename, vocab):
    with open(input_filename) as f:
        for line in f:
            yield sorted(
                ((vocab[t], w) for (t, w) in
                    enumerate([float(w) for w in line.strip().split()])),
                key=lambda p: p[1],
                reverse=True)


def main(trunc_csv, vocab_filename, input_filename, output_filename):
    vocab = utils.load_vocab(vocab_filename)

    node_topics = []
    graph = {}

    for ww_list in sorted_topic_word_weight_lists(input_filename, vocab):
        label = []
        topic_weight = sum(ww[1] for ww in ww_list)
        for (word, weight) in utils.take(ww_list, NUM_TYPES_PER_TOPIC):
            label.append((word, weight))
        node_topics.append((label, topic_weight))

    trunc = [int(t) for t in trunc_csv.split(',')]
    m = utils.tree_index_m(trunc)
    b = utils.tree_index_b(trunc)
    for node in utils.tree_iter(trunc):
        idx = utils.tree_index(node, m, b)
        node_dict = {'words': [{'word': p[0], 'weight': p[1]} for p in node_topics[idx][0]], 'children': [], 'weight': node_topics[idx][1]}
        parent = node[:-1]
        graph[node] = node_dict
        if parent:
            graph[parent]['children'].append(node_dict)

    with open(output_filename, 'w') as f:
        json.dump(graph[(0,)], f, indent=2)


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
