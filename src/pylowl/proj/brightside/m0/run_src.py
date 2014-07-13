#!/usr/bin/env python
#$ -cwd
#$ -j y
#$ -V
#$ -N "m1-src"
#$ -q text.q
#$ -l num_proc=1,mem_free=1G,h_rt=1:00:00


import sys
import os
import shutil
import re
import tempfile
from glob import glob
from pylowl.proj.brightside.m0.run import run
from pylowl.proj.brightside.m0.postproc.generate_d3_graph import generate_d3_graph
from pylowl.proj.brightside.m0.postproc.generate_d3_subgraphs import generate_d3_subgraphs
from pylowl.proj.brightside.preproc.extract_concrete_vocab import extract_concrete_vocab
from pylowl.proj.brightside.preproc.docs_to_concrete import docs_to_concrete


# TODO:
# export PYTHONPATH=build/lib...
# export PYTHONOPTIMIZE=1


SRC_EXTENSIONS = ('.py', '.sh', '.c', '.h', '.pxd', '.pyx')

# paths are relative to littleowl repo root
OUTPUT_DIR_BASE = 'output/pylowl/proj/brightside/m0'
TRUNC = '1,3,2'
POSTPROC_DIR = 'src/pylowl/proj/brightside/postproc'
MY_POSTPROC_DIR = 'src/pylowl/proj/brightside/m0/postproc'



def src_path_filter(path):
    ext = path[path.rfind('.'):]
    return (ext in SRC_EXTENSIONS and 'src/pylowl/' not in path)


def make_data(input_dir, output_dir):
    temp_output_dir = tempfile.mkdtemp()
    i = 0
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if src_path_filter(path):
                shutil.copy(path, os.path.join(temp_output_dir, str(i)))
                i += 1
    docs_to_concrete(os.path.join(temp_output_dir, '*'), output_dir)
    shutil.rmtree(temp_output_dir)


if __name__ == '__main__':
    profile = ('--profile' in sys.argv[1:])

    print 'sys.path:'
    for path in sys.path:
        print '    %s' % path
    print

    print 'os.environ:'
    for (k, v) in os.environ.items():
        print '    %s: %s' % (k, v)
    print

    os.chdir('../../../../..') # repository root

    data_dir = tempfile.mkdtemp()
    make_data('src', data_dir)
    train_data_path = os.path.join(data_dir, '*')
    # TODO would be nice if we didn't test on training data...
    test_data_path = os.path.join(data_dir, '*')

    (fd, vocab_path) = tempfile.mkstemp()
    os.close(fd)
    extract_concrete_vocab(glob(train_data_path), 0, 0, 0, vocab_path)

    print 'Creating output directory...'
    if not os.path.isdir(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE)
    output_dir = tempfile.mkdtemp(dir=OUTPUT_DIR_BASE)

    print 'Running stochastic variational inference...'
    code = '''run(trunc=TRUNC,
        data_path=train_data_path,
        test_data_path=test_data_path,
        test_samples=50,
        init_samples=50,
        batchsize=20,
        max_time=300,
        save_model=True,
        output_dir=output_dir,
        vocab_path=vocab_path,
        log_level='DEBUG')'''.replace('\n', ' ')
    if profile:
        print 'Profiling...'
        import cProfile
        cProfile.run(code, os.path.join(output_dir, 'profile'))
    else:
        exec code

    print 'Generating D3 inputs...'
    generate_d3_graph(output_dir, os.path.join(output_dir, 'graph.json'))
    generate_d3_subgraphs(output_dir, os.path.join(output_dir, 'subgraphs.json'),
                          doc_id_re=re.compile(r'.*/test/\d+\.concrete$'))

    print 'Linking visualization code to output directory...'
    for basename in ('graph.html', 'subgraphs.html'):
        os.symlink(os.path.abspath(os.path.join(MY_POSTPROC_DIR, basename)),
            os.path.join(output_dir, basename))
    for basename in ('d3.v3.js',):
        os.symlink(os.path.abspath(os.path.join(POSTPROC_DIR, basename)),
            os.path.join(output_dir, basename))

    shutil.rmtree(data_dir)

    print 'Done:'
    print output_dir
