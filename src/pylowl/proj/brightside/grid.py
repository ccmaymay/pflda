#!/usr/bin/env python


import subprocess
import itertools as it


TRUNC_LEVELS = (
    '1,7,5,3',
    '1,7,5',
    '1,35',
)

ALPHA_LEVELS = (
    #'2.0',
    '1.0',
    '0.5',
)

BETA_LEVELS = (
    #'2.0',
    '1.0',
    '0.5',
)

GAMMA1_LEVELS = (
    #'2.0',
    '1.0',
    '0.5',
)

GAMMA2_LEVELS = (
    #'2.0',
    '1.0',
    '0.5',
)

KAPPA_LEVELS = (
    '0.5',
)

IOTA_LEVELS = (
    '1.0',
)

BATCHSIZE_LEVELS = (
    '100',
)


def main():
    cells = it.product(TRUNC_LEVELS, ALPHA_LEVELS, BETA_LEVELS,
                       GAMMA1_LEVELS, GAMMA2_LEVELS, KAPPA_LEVELS, IOTA_LEVELS,
                       BATCHSIZE_LEVELS)
    for (trunc, alpha, beta, gamma1, gamma2, kappa, iota, batchsize) in cells:
        args = ['qsub', 'run_m0.qsub', trunc, '--init_samples=1000', '--test_samples=100', '--save_model', '--batchsize', batchsize,'--gamma1', gamma1, '--gamma2', gamma2, '--alpha', alpha, '--beta', beta, '--kappa', kappa, '--iota', iota]
        print ' '.join(args)
        subprocess.call(args)


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
