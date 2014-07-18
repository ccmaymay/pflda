#!/usr/bin/env python
#$ -cwd
#$ -j y
#$ -V
#$ -N "hdp-tng"
#$ -q text.q
#$ -l num_proc=1,mem_free=2G,h_rt=2:00:00

import sys
import os
import re
import tempfile
from pylowl.proj.brightside.hdp.run import run

profile = ('--profile' in sys.argv[1:])

print 'sys.path:'
for path in sys.path:
    print '    %s' % path
print

print 'os.environ:'
for (k, v) in os.environ.items():
    print '    %s: %s' % (k, v)
print

K = 20
L = 10
DATA_DIR = 'data/txt/tng'
VOCAB_PATH = os.path.join(DATA_DIR, 'vocab')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

# TODO
HDP_SRC_DIR = 'src/pylowl/proj/brightside/hdp'
if not os.path.isdir(HDP_SRC_DIR):
    sys.stderr.write('%s does not exist.\n' % HDP_SRC_DIR)
    sys.stderr.write('Postprocessing will fail.\n')
    sys.stderr.write('Note that this script should be run from the littleowl repository root.\n')

print 'Creating output directory...'
OUTPUT_DIR_BASE = 'output/pylowl/proj/brightside/hdp'
if not os.path.isdir(OUTPUT_DIR_BASE):
    os.makedirs(OUTPUT_DIR_BASE)
OUTPUT_DIR = tempfile.mkdtemp(prefix='', suffix='', dir=OUTPUT_DIR_BASE)
umask = os.umask(0o022) # whatever, python
os.umask(umask) # set umask back
os.chmod(OUTPUT_DIR, 0o0755 & ~umask)

print 'Running stochastic variational inference...'
code = '''run(K=K, L=L,
    data_dir=TRAIN_DATA_DIR,
    test_data_dir=TEST_DATA_DIR,
    test_samples=400,
    init_samples=400,
    max_time=360,
    save_model=True,
    output_dir=OUTPUT_DIR,
    vocab_path=VOCAB_PATH,
    log_level='DEBUG')'''.replace('\n', ' ')
if profile:
    print 'Profiling...'
    import cProfile
    cProfile.run(code, os.path.join(OUTPUT_DIR, 'profile'))
else:
    exec code

print 'Done:'
print OUTPUT_DIR
