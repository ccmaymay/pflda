#!/usr/bin/env python 


from pylowl.proj.brightside.utils import seconds_to_hms, parse_grid_args, run_grid_commands
from pylowl.proj.brightside.shdp.run import DEFAULT_OPTIONS, GRID_VAR_NAME_TYPE_PAIRS
from pylowl.proj.brightside.preproc.src_to_concrete import src_to_concrete


(DATA_DIR, TEST_DATA_DIR, VOCAB_PATH) = src_to_concrete('src', 'data/txt')
INIT_SAMPLES = 50
MAX_TIME = 90

COMMAND = (
    'qsub',
    '-l', 'num_proc=1,mem_free=2G,h_rt=%d:%02d:%02d' % seconds_to_hms(3600 + 2 * MAX_TIME),
    'src/pylowl/proj/brightside/shdp/run.py',
    '--data_dir', DATA_DIR,
    '--test_data_dir', TEST_DATA_DIR,
    '--vocab_path', VOCAB_PATH,
    '--init_samples', INIT_SAMPLES,
    '--batchsize', 20,
    '--max_time', MAX_TIME
)


if __name__ == '__main__':
    args = parse_grid_args(GRID_VAR_NAME_TYPE_PAIRS, DEFAULT_OPTIONS)
    run_grid_commands(COMMAND, **args)
