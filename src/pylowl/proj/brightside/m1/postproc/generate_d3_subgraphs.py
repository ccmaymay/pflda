#!/usr/bin/env python


import os
import re
import json
import itertools as it
from datetime import datetime
from pylowl.proj.brightside.corpus import load_concrete_docs, load_concrete_doc
from pylowl.proj.brightside.utils import tree_index_m, tree_index_b, tree_iter, tree_index, load_options, nested_file_paths


EPOCH = datetime(1970, 1, 1)
DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S'


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


def expectation(p, x):
    if x:
        return sum([p_i*x_i for (p_i, x_i) in zip(p, x)])
    else:
        return None


def normalized(weights):
    weights_sum = float(sum(weights))
    return [w/weights_sum for w in weights]


def parse_datetime(datetime_str):
    return datetime.strptime(datetime_str, DATETIME_FORMAT)


def datetime_to_float(d):
    return (d - EPOCH).total_seconds()


def generate_d3_subgraphs(result_dir, output_filename):
    options = load_options(os.path.join(result_dir, 'options'))
    trunc_csv = options['trunc']
    test_data_dir = options['test_data_dir']
    subtree_filename = os.path.join(result_dir, 'final.subtree')
    lambda_ss_filename = os.path.join(result_dir, 'final.subtree_lambda_ss')
    doc_lambda_ss_filename = os.path.join(result_dir, 'final.subtree_doc_lambda_ss')
    Elogpi_filename = os.path.join(result_dir, 'final.subtree_Elogpi')
    logEpi_filename = os.path.join(result_dir, 'final.subtree_logEpi')

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
                    raise Exception('users do not match: %s and %s'
                                    % (user, stat_pieces[0]))
                weights = [float(w) for w in stat_pieces[1:]]
                for node in tree_iter(trunc):
                    idx = tree_index(node, m, b)
                    if idx in node_map:
                        subtree_dicts[idx][stat_name] = weights[node_map[idx]]

            subtree_dicts_per_user[user] = subtree_dicts
            node_maps_per_user[user] = node_map

    doc_id_path_map = dict()
    test_data_paths = nested_file_paths(test_data_dir)
    for doc in load_concrete_docs(test_data_paths):
        doc_id_path_map[doc.id] = doc.path

    weighted_datetimes_per_user = dict(
        (
            user,
            [[] for idx in range(len(subtree_dicts))]
        )
        for user in subtree_dicts_per_user
    )
    with open(doc_lambda_ss_filename) as doc_lambda_ss_f:
        for line in doc_lambda_ss_f:
            pieces = line.strip().split()
            user = pieces[0]
            doc_identifier = pieces[1]
            doc = load_concrete_doc(doc_id_path_map[doc_identifier])
            if 'datetime' in doc.attrs and doc.attrs['datetime'] is not None:
                datetime_float = datetime_to_float(parse_datetime(doc.attrs['datetime']))
                weights = [float(w) for w in pieces[2:]]
                node_map = node_maps_per_user[user]
                for (idx, i) in node_map.items():
                    weighted_datetimes_per_user[user][idx].append((weights[i], datetime_float))

    for (user, subtree_dicts) in subtree_dicts_per_user.items():
        node_map = node_maps_per_user[user]
        for node in tree_iter(trunc):
            idx = tree_index(node, m, b)
            if idx in node_map:
                weighted_datetime_floats = weighted_datetimes_per_user[user][idx]
                weights = [wdf[0] for wdf in weighted_datetime_floats]
                datetime_floats = [wdf[1] for wdf in weighted_datetime_floats]
                if weights:
                    probabilities = normalized(weights)
                    Edatetime = expectation(probabilities, datetime_floats)
                    subtree_dicts[idx]['expected_time'] = Edatetime
                    subtree_dicts[idx]['min_time'] = min(datetime_floats)
                    subtree_dicts[idx]['max_time'] = max(datetime_floats)

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
