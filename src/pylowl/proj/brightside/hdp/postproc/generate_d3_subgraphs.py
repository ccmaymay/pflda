#!/usr/bin/env python


import os
import re
import json
import itertools as it
from pylowl.proj.brightside.utils import load_options


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
    K = int(options['K'])
    L = int(options['L'])
    sublist_filename = os.path.join(result_dir, 'sublist')
    lambda_ss_filename = os.path.join(result_dir, 'sublist_lambda_ss')
    Elogpi_filename = os.path.join(result_dir, 'sublist_Elogpi')
    logEpi_filename = os.path.join(result_dir, 'sublist_logEpi')

    sublist_dicts_per_doc = {}

    with open(sublist_filename) as sublist_f, \
         open(Elogpi_filename) as Elogpi_f, \
         open(logEpi_filename) as logEpi_f, \
         open(lambda_ss_filename) as lambda_ss_f:
        for (sublist_line, stat_lines) in it.izip(sublist_f, it.izip(
                Elogpi_f, logEpi_f, lambda_ss_f)):
            pieces = sublist_line.strip().split()
            doc_id = pieces[0]
            phi = [float(z) for z in pieces[1:]]

            sublist_dicts = [None] * L
            for j in xrange(L):
                sublist_dicts[j] = {
                    'phi': phi[j*K:(j+1)*K],
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
                for j in xrange(L):
                    sublist_dicts[j][stat_name] = weights[j]

            sublist_dicts_per_doc[doc_id] = sublist_dicts

    json_data = []

    for (doc_id, sublist_dicts) in sublist_dicts_per_doc.items():
        lambda_ss_sum = 0
        for j in xrange(L):
            if j > 0:
                sublist_dicts[j-1]['children'].append(sublist_dicts[j])
            lambda_ss_sum += sublist_dicts[j]['lambda_ss']
        json_data.append({
            'doc_id': doc_id,
            'sublist': sublist_dicts[0],
            'lambda_ss_sum': lambda_ss_sum
        })

    json_data.sort(key=lambda s: s['lambda_ss_sum'], reverse=True)

    with open(output_filename, 'w') as f:
        json.dump(json_data, f, indent=2)


if __name__ == '__main__':
    main()
