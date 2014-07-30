#!/usr/bin/env python 


from pylowl.proj.brightside.utils import seconds_to_hms, parse_grid_args, run_grid_commands
from pylowl.proj.brightside.m0.run import DEFAULT_OPTIONS


DATA_DIR = 'data/txt/ds2/train'
TEST_DATA_DIR = 'data/txt/ds2/test'
VOCAB_PATH = 'data/txt/ds2/vocab'
TEST_SAMPLES = 1000
INIT_SAMPLES = 1000
MAX_TIME = 360

COMMAND = (
    'qsub', 'src/pylowl/proj/brightside/m0/run.py',
    '-l', 'num_proc=1,mem_free=2G,h_rt=%d:%02d:%02d' % seconds_to_hms(3600 + 2 * MAX_TIME),
    '--',
    '--data_dir', DATA_DIR,
    '--test_data_dir', TEST_DATA_DIR,
    '--test_samples', TEST_SAMPLES,
    '--init_samples', INIT_SAMPLES,
    '--U', 9474,
    '--D', 210531,
    '--streaming',
    '--user_doc_reservoir_capacity', 1000,
    '--user_subtree_selection_interval', 100,
    '--max_time', MAX_TIME
)


VARS = (
    ('trunc', str),
    ('alpha', float),
    ('beta', float),
    ('gamma1', float),
    ('gamma2', float),
    ('lambda0', float),
    ('kappa', float),
)


if __name__ == '__main__':
    args = parse_grid_args(VAR_NAME_TYPE_PAIRS, DEFAULT_OPTIONS)
    run_grid_commands(COMMAND, **args)
