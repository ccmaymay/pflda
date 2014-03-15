#!/usr/bin/env python


import os.path
import sys
import re


DEFAULT_ITER_KEY = 'doc'

PER_ITER_STAT_NAMES = (
    'num words',
    'init in-sample nmi',
    'in-sample nmi',
    'out-of-sample nmi',
    'out-of-sample perplexity',
    'out-of-sample log-likelihood',
    'out-of-sample coherence',
)

PER_ITER_COUNT_NAMES = (
    'resampling',
)

BADNESS_RE = re.compile(r'\W')


def normalize_stat_name(name):
    return BADNESS_RE.subn('_', name.lower())[0]


def dict_key(iter_num, key):
    return (normalize_stat_name(key), iter_num)


def dict_stat_get(d, iter_num, key):
    return d.get(dict_key(iter_num, key), 'NA')


def dict_count_get(d, iter_num, key):
    return d.get(dict_key(iter_num, key), 0)


def dict_set(d, iter_num, key, val):
    d[dict_key(iter_num, key)] = val


def dict_increment(d, iter_num, key):
    k = dict_key(iter_num, key)
    if k in d:
        d[k] += 1
    else:
        d[k] = 0


def process_logs(experiment_path, iter_key=None):
    if iter_key is None:
        iter_key = DEFAULT_ITER_KEY

    if not os.path.isdir(experiment_path):
        raise Exception(experiment_path + ' is not a directory')

    for dataset_entry in os.listdir(experiment_path):
        dataset_path = os.path.join(experiment_path, dataset_entry)
        if os.path.isdir(dataset_path):
            iter_num_bound = 0
            per_iter_stats_list = []
            entries = os.listdir(dataset_path)
            entries.sort()

            for entry in entries:
                path = os.path.join(dataset_path, entry)
                if os.path.isfile(path) and not entry.startswith('.'):
                    (last_iter_num, per_iter_stats) = parse_log(path, iter_key)
                    if last_iter_num is not None and per_iter_stats:
                        iter_num_bound = max(last_iter_num + 1, iter_num_bound)
                        per_iter_stats_list.append(per_iter_stats)

            for stat_name in PER_ITER_STAT_NAMES:
                dataset_tab_path = dataset_path + '_%s.tab' % normalize_stat_name(stat_name)
                with open(dataset_tab_path, 'w') as f:
                    f.write('\t'.join(['iter'] + ['run.%d' % i for i in range(len(per_iter_stats_list))]) + '\n')
                    for iter_num in range(iter_num_bound):
                        vals = [str(dict_stat_get(per_iter_stats, iter_num, stat_name)) for per_iter_stats in per_iter_stats_list]
                        if vals.count('NA') < len(vals):
                            f.write('\t'.join([str(iter_num)] + vals) + '\n')

            for count_name in PER_ITER_COUNT_NAMES:
                dataset_tab_path = dataset_path + '_%s.tab' % normalize_stat_name(count_name)
                with open(dataset_tab_path, 'w') as f:
                    f.write('\t'.join(['iter'] + ['run.%d' % i for i in range(len(per_iter_stats_list))]) + '\n')
                    for iter_num in range(iter_num_bound):
                        vals = [str(dict_count_get(per_iter_stats, iter_num, count_name)) for per_iter_stats in per_iter_stats_list]
                        if vals.count('0') < len(vals):
                            f.write('\t'.join([str(iter_num)] + vals) + '\n')


def parse_log(log_filename, iter_key):
    iter_num = None
    per_iter_stats = dict()

    with open(log_filename) as f:
        for line in f:
            split_idx = line.find(':')
            if split_idx >= 0:
                key = line[:split_idx]
                val = line[split_idx+1:].strip()
            else:
                key = None
                val = None

            if key == iter_key:
                iter_num = int(val)
            elif key in PER_ITER_STAT_NAMES:
                dict_set(per_iter_stats, iter_num, key, val)
            elif key in PER_ITER_COUNT_NAMES:
                dict_increment(per_iter_stats, iter_num, key)

    return (iter_num, per_iter_stats)


if __name__ == '__main__':
    process_logs(*sys.argv[1:])
