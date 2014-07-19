#!/usr/bin/env python
#$ -cwd
#$ -j y
#$ -V
#$ -N "hdp-src"
#$ -q text.q
#$ -l num_proc=1,mem_free=1G,h_rt=1:00:00


import sys
import os
import shutil
import re
import tempfile
from pylowl.proj.brightside.corpus import write_concrete_doc, Document
from pylowl.proj.brightside.utils import nested_file_paths
from pylowl.proj.brightside.hdp.run import run
from pylowl.proj.brightside.preproc.extract_concrete_vocab import extract_concrete_vocab


# TODO:
# export PYTHONPATH=build/lib...
# export PYTHONOPTIMIZE=1


SPLIT_RE = re.compile(r'\W+')
SRC_EXTENSIONS = ('.py', '.sh', '.c', '.h', '.pxd', '.pyx')

# paths are relative to littleowl repo root
OUTPUT_DIR_BASE = 'output/pylowl/proj/brightside/hdp'
K = 20
L = 10
POSTPROC_DIR = 'src/pylowl/proj/brightside/postproc'
MY_POSTPROC_DIR = 'src/pylowl/proj/brightside/hdp/postproc'


if not os.path.isdir(POSTPROC_DIR):
    sys.stderr.write('%s does not exist.\n' % POSTPROC_DIR)
    sys.stderr.write('Postprocessing will fail.\n')
    sys.stderr.write('Note that this script should be run from the littleowl repository root.\n')


def src_path_filter(path):
    ext = path[path.rfind('.'):]
    return (ext in SRC_EXTENSIONS
            and not (ext == '.c' and 'src/pylowl/' in path))


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def make_data(input_dir, output_dir):
    i = 0
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if src_path_filter(path):
                tokens = []
                with open(path) as f:
                    for line in f:
                        tokens.extend(token for token in SPLIT_RE.split(line)
                                      if token and not is_num(token))
                if tokens:
                    output_path = os.path.join(output_dir, '%d.concrete' % i)
                    write_concrete_doc(Document(tokens, id=path), output_path)
                    i += 1


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

    data_dir = tempfile.mkdtemp()
    umask = os.umask(0o022) # whatever, python
    os.umask(umask) # set umask back
    os.chmod(data_dir, 0o0755 & ~umask)
    make_data('src', data_dir)
    train_data_dir = data_dir
    # TODO would be nice if we didn't test on training data...
    test_data_dir = data_dir

    (fd, vocab_path) = tempfile.mkstemp()
    os.close(fd)
    extract_concrete_vocab(nested_file_paths(train_data_dir), 0, 0, 0, vocab_path)

    print 'Creating output directory...'
    if not os.path.isdir(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE)
    output_dir = tempfile.mkdtemp(dir=OUTPUT_DIR_BASE, prefix='')
    os.chmod(output_dir, 0o0755 & ~umask)

    print 'Running stochastic variational inference...'
    code = '''run(K=K, L=L,
        data_dir=train_data_dir,
        test_data_dir=test_data_dir,
        test_samples=50,
        init_samples=50,
        batchsize=20,
        max_time=90,
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

    print 'Linking visualization code to output directory...'
    for basename in ('subgraphs.html',):
        os.symlink(os.path.abspath(os.path.join(MY_POSTPROC_DIR, basename)),
            os.path.join(output_dir, basename))
    for basename in ('d3.v3.js', 'core.js', 'graph.html'):
        os.symlink(os.path.abspath(os.path.join(POSTPROC_DIR, basename)),
            os.path.join(output_dir, basename))

    shutil.rmtree(data_dir)

    print 'Done:'
    print output_dir
