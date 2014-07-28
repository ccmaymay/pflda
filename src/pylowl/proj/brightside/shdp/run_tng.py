#!/usr/bin/env python
#$ -cwd
#$ -j y
#$ -V
#$ -N "shdp-tng"
#$ -q text.q
#$ -l num_proc=1,mem_free=2G,h_rt=2:00:00

import sys
import os
import re
import shutil
import pkg_resources
import tempfile
from pylowl.proj.brightside.shdp.run import run
from pylowl.proj.brightside.shdp.postproc.generate_d3_graph import generate_d3_graph
from pylowl.proj.brightside.shdp.postproc.generate_d3_subgraphs import generate_d3_subgraphs


POSTPROC_PKG = 'pylowl.proj.brightside.shdp.postproc'
BRIGHTSIDE_POSTPROC_PKG = 'pylowl.proj.brightside.postproc'

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
J = 10
I = 5
M = 2
DATA_DIR = 'data/txt/tng'
VOCAB_PATH = os.path.join(DATA_DIR, 'vocab')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

print 'Creating output directory...'
OUTPUT_DIR_BASE = 'output/pylowl/proj/brightside/shdp'
if not os.path.isdir(OUTPUT_DIR_BASE):
    os.makedirs(OUTPUT_DIR_BASE)
OUTPUT_DIR = tempfile.mkdtemp(prefix='', suffix='', dir=OUTPUT_DIR_BASE)
umask = os.umask(0o022) # whatever, python
os.umask(umask) # set umask back
os.chmod(OUTPUT_DIR, 0o0755 & ~umask)

print 'Running stochastic variational inference...'
code = '''run(I=I, J=J, K=K,
    data_dir=TRAIN_DATA_DIR,
    test_data_dir=TEST_DATA_DIR,
    test_samples=300,
    init_samples=300,
    alpha=1,
    beta=1,
    gamma=1,
    iota=64,
    kappa=0.6,
    lambda0=0.005,
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

print 'Generating D3 inputs...'
generate_d3_graph(OUTPUT_DIR, os.path.join(OUTPUT_DIR, 'graph.json'))
generate_d3_subgraphs(OUTPUT_DIR, os.path.join(OUTPUT_DIR, 'subgraphs.json'))

print 'Linking visualization code to output directory...'
for basename in ('subgraphs.html',):
    shutil.copy(pkg_resources.resource_filename(POSTPROC_PKG, basename),
        os.path.join(OUTPUT_DIR, basename))
for basename in ('d3.v3.js', 'core.js', 'graph.html'):
    shutil.copy(pkg_resources.resource_filename(BRIGHTSIDE_POSTPROC_PKG, basename),
        os.path.join(OUTPUT_DIR, basename))

print 'Done:'
print OUTPUT_DIR
