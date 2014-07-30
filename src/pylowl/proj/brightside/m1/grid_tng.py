#!/usr/bin/env python 


from pylowl.proj.brightside.utils import seconds_to_hms, parse_grid_args, run_grid_commands
from pylowl.proj.brightside.m1.run import DEFAULT_OPTIONS, GRID_VAR_NAME_TYPE_PAIRS


DATA_DIR = 'data/txt/tng/train'
TEST_DATA_DIR = 'data/txt/tng/test'
VOCAB_PATH = 'data/txt/tng/vocab'
TEST_SAMPLES = 400
INIT_SAMPLES = 400
MAX_TIME = 360

COMMAND = (
    'qsub',
    '-l', 'num_proc=1,mem_free=2G,h_rt=%d:%02d:%02d' % seconds_to_hms(3600 + 2 * MAX_TIME),
    'src/pylowl/proj/brightside/m1/run.py',
    '--data_dir', DATA_DIR,
    '--test_data_dir', TEST_DATA_DIR,
    '--vocab_path', VOCAB_PATH,
    '--test_samples', TEST_SAMPLES,
    '--init_samples', INIT_SAMPLES,
    '--max_time', MAX_TIME
)


if __name__ == '__main__':
    args = parse_grid_args(GRID_VAR_NAME_TYPE_PAIRS, DEFAULT_OPTIONS)
    run_grid_commands(COMMAND, **args)
