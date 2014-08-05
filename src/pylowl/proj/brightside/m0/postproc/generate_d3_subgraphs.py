#!/usr/bin/env python


import os
import json
import itertools as it
from pylowl.proj.brightside.utils import tree_index_m, tree_index_b, tree_iter, tree_index, load_options, nested_file_paths
from pylowl.proj.brightside.corpus import load_concrete_docs, load_concrete_doc


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_dir', type=str,
                        help='path to dir where model was saved')
    parser.add_argument('output_path', type=str,
                        help='output file path')

    args = parser.parse_args()
    generate_d3_subgraphs(
        args.result_dir,
        args.output_path,
    )


def generate_d3_subgraphs(result_dir, output_filename):
    options = load_options(os.path.join(result_dir, 'options'))
    test_data_dir = options['test_data_dir']
    trunc_csv = options['trunc']
    subtree_filename = os.path.join(result_dir, 'subtree')
    lambda_ss_filename = os.path.join(result_dir, 'subtree_lambda_ss')
    Elogpi_filename = os.path.join(result_dir, 'subtree_Elogpi')
    logEpi_filename = os.path.join(result_dir, 'subtree_logEpi')

    trunc = [int(t) for t in trunc_csv.split(',')]
    m = tree_index_m(trunc)
    b = tree_index_b(trunc)

    subtree_dicts_per_id = {}

    doc_id_class_map = dict()
    test_data_paths = nested_file_paths(test_data_dir)
    for doc in load_concrete_docs(test_data_paths):
        doc_id_class_map[doc.id] = doc.attrs.get('class', None)

    with open(subtree_filename) as subtree_f, \
         open(Elogpi_filename) as Elogpi_f, \
         open(logEpi_filename) as logEpi_f, \
         open(lambda_ss_filename) as lambda_ss_f:
        for (subtree_line, stat_lines) in it.izip(subtree_f, it.izip(
                Elogpi_f, logEpi_f, lambda_ss_f)):
            pieces = subtree_line.strip().split()
            doc_id = pieces[0]
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
                if doc_id != stat_pieces[0]:
                    raise Exception('doc ids do not match: %s and %s'
                                    % (doc_id, stat_pieces[0]))
                weights = [float(w) for w in stat_pieces[1:]]
                for node in tree_iter(trunc):
                    idx = tree_index(node, m, b)
                    if idx in node_map:
                        subtree_dicts[idx][stat_name] = weights[node_map[idx]]

            subtree_dicts_per_id[doc_id] = subtree_dicts

    json_data = []

    for (doc_id, subtree_dicts) in subtree_dicts_per_id.items():
        num_active = 0
        for node in tree_iter(trunc):
            idx = tree_index(node, m, b)
            p = node[:-1]
            num_active += subtree_dicts[idx]['active']
            if p:
                p_idx = tree_index(p, m, b)
                subtree_dicts[p_idx]['children'].append(subtree_dicts[idx])
        json_data.append({
            'doc_id': doc_id,
            'subtree': subtree_dicts[0],
            'class': doc_id_class_map[doc_id],
            'num_active': num_active
        })

    json_data.sort(key=lambda s: s['num_active'], reverse=True)

    with open(output_filename, 'w') as f:
        json.dump(json_data, f, indent=2)


if __name__ == '__main__':
    main()
