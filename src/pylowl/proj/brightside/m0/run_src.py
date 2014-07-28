#!/usr/bin/env python
#$ -cwd
#$ -j y
#$ -V
#$ -N "m0-src"
#$ -q text.q
#$ -l num_proc=1,mem_free=1G,h_rt=1:00:00


if __name__ == '__main__':
    import sys
    import os

    print 'sys.path:'
    for path in sys.path:
        print '    %s' % path
    print

    print 'os.environ:'
    for (k, v) in os.environ.items():
        print '    %s: %s' % (k, v)
    print


    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(src_dir='src')
    parser.add_argument('--src_dir', type=str, required=False,
                        help='path to littleowl source tree')
    args = parser.parse_args()

    import shutil
    from pylowl.proj.brightside.utils import make_output_dir
    from pylowl.proj.brightside.preproc.src_to_concrete import src_to_concrete
    from pylowl.proj.brightside.m0.run import run
    from pylowl.proj.brightside.m0.postproc.utils import postprocess

    print 'Creating data from source tree...'
    (train_data_dir, test_data_dir, vocab_path) = src_to_concrete(args.src_dir)

    print 'Creating output directory...'
    output_dir = make_output_dir('output/pylowl/proj/brightside/m0')

    print 'Running stochastic variational inference...'
    run(trunc='1,3,2',
        data_dir=train_data_dir,
        test_data_dir=test_data_dir,
        test_samples=50,
        init_samples=50,
        batchsize=20,
        alpha=1,
        beta=1,
        gamma1=1,
        gamma2=1,
        iota=64,
        kappa=0.6,
        lambda0=0.01,
        max_time=90,
        save_model=True,
        output_dir=output_dir,
        vocab_path=vocab_path,
        log_level='DEBUG')

    print 'Postprocessing...'
    postprocess(output_dir)

    shutil.rmtree(train_data_dir, ignore_errors=True)
    shutil.rmtree(test_data_dir, ignore_errors=True)
    try:
        os.remove(vocab_path)
    except IOError:
        pass

    print 'Done:'
    print output_dir
