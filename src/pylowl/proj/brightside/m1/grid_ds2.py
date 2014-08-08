#!/usr/bin/env python 


from pylowl.proj.brightside.utils import seconds_to_hms, parse_grid_args, run_grid_commands
from pylowl.proj.brightside.m0.run import DEFAULT_OPTIONS, GRID_VAR_NAME_TYPE_PAIRS
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
arg_parser.set_defaults(data_dir='data/txt/ds2/train',
                        test_data_dir='data/txt/ds2/test',
                        vocab_path='data/txt/ds2/vocab',
                        batchsize=100, init_samples=1000, test_samples=1000,
                        user_doc_reservoir_capacity=10,
                        user_subtree_selection_interval=10,
                        U=9474, D=210531,
                        max_time=4*3600)
arg_parser.add_argument('--data_dir', type=str)
arg_parser.add_argument('--test_data_dir', type=str)
arg_parser.add_argument('--vocab_path', type=str)
arg_parser.add_argument('--init_samples', type=int)
arg_parser.add_argument('--test_samples', type=int)
arg_parser.add_argument('--batchsize', type=int)
arg_parser.add_argument('--max_time', type=int)
arg_parser.add_argument('--user_doc_reservoir_capacity', type=int)
arg_parser.add_argument('--user_subtree_selection_interval', type=int)
arg_parser.add_argument('--U', type=int)
arg_parser.add_argument('--D', type=int)
(args, passthru_args) = arg_parser.parse_known_args()

command = (
    'qsub',
    '-l', 'num_proc=1,mem_free=2G,h_rt=%d:%02d:%02d' % seconds_to_hms(3600 + 2 * args.max_time),
    'src/pylowl/proj/brightside/m0/run.py',
    '--streaming',
    '--U', args.U,
    '--D', args.D,
    '--data_dir', args.data_dir,
    '--test_data_dir', args.test_data_dir,
    '--vocab_path', args.vocab_path,
    '--init_samples', args.init_samples,
    '--test_samples', args.test_samples,
    '--batchsize', args.batchsize,
    '--max_time', args.max_time,
    '--user_doc_reservoir_capacity', args.user_doc_reservoir_capacity,
    '--user_subtree_selection_interval', args.user_subtree_selection_interval,
)


if __name__ == '__main__':
    args = parse_grid_args(GRID_VAR_NAME_TYPE_PAIRS, DEFAULT_OPTIONS, passthru_args)
    run_grid_commands(command, **args)
