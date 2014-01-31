#!/usr/bin/env python


import os
import shutil


DATASETS = ('diff3', 'sim3', 'rel3')
EXPERIMENTS = (
    '0-rs0', '0-rs100', '0-rs1000', '0-rs10000', '0-rs100000', '0-rs500000',
    '1-rs0', '1-rs100', '1-rs1000', '1-rs10000', '1-rs100000', '1-rs500000',
)


def main():
    unique_run_stems = list(set(
        [s[:s.rfind('.')] for s in os.listdir('.') if s.startswith('slda.o')]))
    unique_run_stems.sort()

    entry_dict = dict()
    entries = os.listdir('.')
    entries.sort()
    for entry in entries:
        if entry.startswith('slda.o'):
            run_stem = entry[:entry.rfind('.')]
            if run_stem in entry_dict:
                entry_dict[run_stem].append(entry)
            else:
                entry_dict[run_stem] = [entry]

    run_stems = list(entry_dict.keys())
    run_stems.sort()

    i = 0
    for dataset in DATASETS:
        for experiment in EXPERIMENTS:
            run_stem = run_stems[i]
            entries = entry_dict[run_stem]
            print('%s %s' % (dataset, experiment))
            d = os.path.join(experiment, dataset)
            os.makedirs(d)
            for entry in entries:
                print('\t%s' % entry)
                shutil.copy(entry, d)
            i += 1

    if i != len(run_stems):
        raise Exception('i is %d but there are %d run stems'
            % (i, len(run_stems)))


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
