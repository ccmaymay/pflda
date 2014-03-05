#!/usr/bin/env python


import os.path
import sys
import re


PER_DOC_STAT_NAMES = (
    'num words',
    'init in-sample nmi',
    'in-sample nmi',
    'out-of-sample nmi',
    'out-of-sample perplexity',
    'out-of-sample log-likelihood',
    'out-of-sample coherence',
)

PER_DOC_COUNT_NAMES = (
    'resampling',
)

BADNESS_RE = re.compile(r'\W')


def normalize_stat_name(name):
    return BADNESS_RE.subn('_', name.lower())[0]


def dict_key(doc_num, key):
    return (normalize_stat_name(key), doc_num)


def dict_stat_get(d, doc_num, key):
    return d.get(dict_key(doc_num, key), 'NA')


def dict_count_get(d, doc_num, key):
    return d.get(dict_key(doc_num, key), 0)


def dict_set(d, doc_num, key, val):
    d[dict_key(doc_num, key)] = val


def dict_increment(d, doc_num, key):
    k = dict_key(doc_num, key)
    if k in d:
        d[k] += 1
    else:
        d[k] = 0


def process_logs(experiment_path):
    if not os.path.isdir(experiment_path):
        raise Exception(experiment_path + ' is not a directory')

    for dataset_entry in os.listdir(experiment_path):
        dataset_path = os.path.join(experiment_path, dataset_entry)
        if os.path.isdir(dataset_path):
            doc_num_bound = 0
            per_doc_stats_list = []
            entries = os.listdir(dataset_path)
            entries.sort()

            for entry in entries:
                path = os.path.join(dataset_path, entry)
                if os.path.isfile(path) and not entry.startswith('.'):
                    (last_doc_num, per_doc_stats) = parse_log(path)
                    if last_doc_num is not None and per_doc_stats:
                        doc_num_bound = max(last_doc_num + 1, doc_num_bound)
                        per_doc_stats_list.append(per_doc_stats)

            for stat_name in PER_DOC_STAT_NAMES:
                dataset_tab_path = dataset_path + '_%s.tab' % normalize_stat_name(stat_name)
                with open(dataset_tab_path, 'w') as f:
                    f.write('\t'.join(['run.%d' % i for i in range(len(per_doc_stats_list))]) + '\n')
                    for doc_num in range(doc_num_bound):
                        f.write('\t'.join([str(dict_stat_get(per_doc_stats, doc_num, stat_name)) for per_doc_stats in per_doc_stats_list]) + '\n')

            for count_name in PER_DOC_COUNT_NAMES:
                dataset_tab_path = dataset_path + '_%s.tab' % normalize_stat_name(count_name)
                with open(dataset_tab_path, 'w') as f:
                    f.write('\t'.join(['run.%d' % i for i in range(len(per_doc_stats_list))]) + '\n')
                    for doc_num in range(doc_num_bound):
                        f.write('\t'.join([str(dict_count_get(per_doc_stats, doc_num, count_name)) for per_doc_stats in per_doc_stats_list]) + '\n')


def parse_log(log_filename):
    doc_num = 0
    per_doc_stats = dict()

    with open(log_filename) as f:
        for line in f:
            split_idx = line.find(':')
            if split_idx >= 0:
                key = line[:split_idx]
                val = line[split_idx+1:].strip()
            else:
                key = None
                val = None

            if key == 'in-sample nmi':
                doc_num += 1
            if key in PER_DOC_STAT_NAMES:
                dict_set(per_doc_stats, doc_num, key, val)
            elif key in PER_DOC_COUNT_NAMES:
                dict_increment(per_doc_stats, doc_num, key)

    return (doc_num, per_doc_stats)


if __name__ == '__main__':
    process_logs(*sys.argv[1:])
