#!/usr/bin/env python


import re
import json
from pylowl.proj.brightside.utils import tree_index_m, tree_index_b, tree_iter, tree_index


TRUNC_RE = re.compile(r'.* trunc: ([0-9,]+)')
SUBTREE_RE = re.compile(r'.* Subtree global node ids for (.+?): ([0-9 ]+)')


def main(log_filename, output_filename):
    subtrees = dict()
    trunc = None
    tree_idx_m = None
    tree_idx_b = None

    with open(log_filename) as f:
        for line in f:
            m = TRUNC_RE.match(line)
            if m is not None:
                trunc = tuple([int(i) for i in m.group(1).split(',')])

            m = SUBTREE_RE.match(line)
            if m is not None:
                identifier = m.group(1)
                node_ids = set([int(i) for i in m.group(2).split()])
                subtrees[identifier] = node_ids

    json_data = []

    tree_idx_b = tree_index_b(trunc)
    tree_idx_m = tree_index_m(trunc)
    for (identifier, node_ids) in subtrees.items():
        d = dict(identifier=identifier)
        subtree_nodes = [None] * len(list(tree_iter(trunc)))

        for node in tree_iter(trunc):
            idx = tree_index(node, tree_idx_m, tree_idx_b)
            active = idx in node_ids
            subtree_nodes[idx] = dict(active=active, children=[])
            p = node[:-1]
            if p:
                p_idx = tree_index(p, tree_idx_m, tree_idx_b)
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
