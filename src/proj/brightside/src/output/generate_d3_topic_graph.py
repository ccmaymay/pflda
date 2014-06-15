#!/usr/bin/env python


import utils
import os
import math
import json


NUM_TYPES_PER_TOPIC = 10


def main(trunc_csv, input_filename, output_filename):
    trunc = [int(t) for t in trunc_csv.split(',')]

    node_topics = []
    graph = {}
    with open(input_filename) as f:
        for line in f:
            label = []
            pieces = line.strip().split()
            topic_weight = sum(float(w) for w in pieces[1::2])
            types = pieces[0:(2*NUM_TYPES_PER_TOPIC):2]
            weights = [float(w) for w in pieces[1:(2*NUM_TYPES_PER_TOPIC):2]]
            for (t, weight) in zip(types, weights):
                label.append((t, weight))
            node_topics.append((label, topic_weight))

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
