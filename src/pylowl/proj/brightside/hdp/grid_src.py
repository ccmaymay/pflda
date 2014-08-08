#!/usr/bin/env python 


from pylowl.proj.brightside.utils import seconds_to_hms, parse_grid_args, run_grid_commands
from pylowl.proj.brightside.hdp.run import DEFAULT_OPTIONS, GRID_VAR_NAME_TYPE_PAIRS
from pylowl.proj.brightside.preproc.src_to_concrete import src_to_concrete
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


arg_parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
arg_parser.set_defaults(src_dir='src', data_parent_dir='data/txt',
                        batchsize=20, init_samples=50, max_time=90)
arg_parser.add_argument('--src_dir', type=str)
arg_parser.add_argument('--data_parent_dir', type=str)
arg_parser.add_argument('--init_samples', type=int)
arg_parser.add_argument('--batchsize', type=int)
arg_parser.add_argument('--max_time', type=int)
(args, passthru_args) = arg_parser.parse_known_args()

(data_dir, test_data_dir, vocab_path) = src_to_concrete(args.src_dir,
                                                        args.data_parent_dir)

command = (
    'qsub',
    '-l', 'num_proc=1,mem_free=2G,h_rt=%d:%02d:%02d' % seconds_to_hms(3600 + 2 * args.max_time),
    'src/pylowl/proj/brightside/hdp/run.py',
    '--data_dir', data_dir,
    '--test_data_dir', test_data_dir,
    '--vocab_path', vocab_path,
    '--init_samples', args.init_samples,
    '--batchsize', args.batchsize,
    '--max_time', args.max_time,
)


if __name__ == '__main__':
    args = parse_grid_args(GRID_VAR_NAME_TYPE_PAIRS, DEFAULT_OPTIONS, passthru_args)
    run_grid_commands(command, **args)
