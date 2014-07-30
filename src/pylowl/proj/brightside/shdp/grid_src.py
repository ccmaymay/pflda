#!/usr/bin/env python 


from pylowl.proj.brightside.utils import seconds_to_hms, parse_grid_args, run_grid_commands
from pylowl.proj.brightside.shdp.run import DEFAULT_OPTIONS
from pylowl.proj.brightside.preproc.src_to_concrete import src_to_concrete


(DATA_DIR, TEST_DATA_DIR, VOCAB_PATH) = src_to_concrete('src', 'data/txt')
TEST_SAMPLES = 50
INIT_SAMPLES = 50
MAX_TIME = 90

COMMAND = (
    'qsub',
    '-l', 'num_proc=1,mem_free=2G,h_rt=%d:%02d:%02d' % seconds_to_hms(3600 + 2 * MAX_TIME),
    'src/pylowl/proj/brightside/shdp/run.py',
    '--data_dir', DATA_DIR,
    '--test_data_dir', TEST_DATA_DIR,
    '--vocab_path', VOCAB_PATH,
    '--test_samples', TEST_SAMPLES,
    '--init_samples', INIT_SAMPLES,
    '--batchsize', 20,
    '--max_time', MAX_TIME
)


VAR_NAME_TYPE_PAIRS = (
    ('K', int),
    ('J', int),
    ('I', int),
    ('alpha', float),
    ('beta', float),
    ('gamma', float),
    ('lambda0', float),
    ('kappa', float),
)


if __name__ == '__main__':
    args = parse_grid_args(VAR_NAME_TYPE_PAIRS, DEFAULT_OPTIONS)
    run_grid_commands(COMMAND, **args)