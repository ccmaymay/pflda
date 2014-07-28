#!/usr/bin/env python
#$ -cwd
#$ -j y
#$ -V
#$ -N "m1-tng"
#$ -q text.q
#$ -l num_proc=1,mem_free=2G,h_rt=2:00:00


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


    from pylowl.proj.brightside.utils import make_output_dir
    from pylowl.proj.brightside.m1.run import run
    from pylowl.proj.brightside.m1.postproc.utils import postprocess

    print 'Creating output directory...'
    output_dir = make_output_dir('output/pylowl/proj/brightside/m1')

    print 'Running stochastic variational inference...'
    run(trunc='1,5,4',
        data_dir='data/txt/tng/train',
        test_data_dir='data/txt/tng/test',
        test_samples=400,
        init_samples=400,
        alpha=1,
        beta=1,
        gamma1=1,
        gamma2=1,
        iota=64,
        kappa=0.6,
        lambda0=0.005,
        max_time=360,
        save_model=True,
        output_dir=output_dir,
        vocab_path='data/txt/tng/vocab',
        user_doc_reservoir_capacity=200,
        user_subtree_selection_interval=50,
        log_level='INFO')

    print 'Postprocessing...'
    postprocess(output_dir)

    print 'Done:'
    print output_dir
