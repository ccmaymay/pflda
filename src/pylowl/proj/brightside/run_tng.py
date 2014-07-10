#!/usr/bin/env python
#$ -cwd
#$ -j y
#$ -V
#$ -N "m1-tng"
#$ -q text.q
#$ -l num_proc=1,mem_free=2G,h_rt=2:00:00

import sys
import os
import tempfile
from pylowl.proj.brightside.run import run
from pylowl.proj.brightside.postproc.generate_d3_topic_graph import generate_d3_topic_graph
from pylowl.proj.brightside.postproc.generate_d3_subgraphs import generate_d3_subgraphs

profile = ('--profile' in sys.argv[1:])

print 'sys.path:'
for path in sys.path:
    print '    %s' % path
print

print 'os.environ:'
for (k, v) in os.environ.items():
    print '    %s: %s' % (k, v)
print

os.chdir('../../../..') # repository root

TRUNC = '1,5,4'
DATA_DIR = 'data/txt/tng.rasp.concrete.catuser'
POSTPROC_DIR = 'src/pylowl/proj/brightside/postproc'
VOCAB_PATH = os.path.join(DATA_DIR, 'vocab')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train/*')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test/*')

print 'Creating output directory...'
OUTPUT_DIR = tempfile.mkdtemp(prefix='', suffix='',
    dir='output/pylowl/proj/brightside')

print 'Running stochastic variational inference...'
code = '''run(trunc=TRUNC,
    data_path=TRAIN_DATA_PATH,
    test_data_path=TEST_DATA_PATH,
    test_samples=400,
    init_samples=400,
    max_time=3600,
    save_model=True,
    output_dir=OUTPUT_DIR,
    concrete=True,
    concrete_vocab_path=VOCAB_PATH,
    U=20,
    D=11222,
    W=4571,
    streaming=True,
    user_doc_reservoir_capacity=200,
    user_subtree_selection_interval=50,
    log_level='DEBUG')'''.replace('\n', ' ')
if profile:
    import cProfile
    cProfile.run(code, os.path.join(OUTPUT_DIR, 'profile'))
else:
    exec code

print 'Generating D3 inputs...'
generate_d3_topic_graph(TRUNC, VOCAB_PATH,
    os.path.join(OUTPUT_DIR, 'final.lambda_ss'),
    os.path.join(OUTPUT_DIR, 'final.Elogpi'),
    os.path.join(OUTPUT_DIR, 'final.logEpi'),
    os.path.join(OUTPUT_DIR, 'final.Elogtheta'),
    os.path.join(OUTPUT_DIR, 'final.logEtheta'),
    os.path.join(OUTPUT_DIR, 'graph.json'))
generate_d3_subgraphs(TRUNC, VOCAB_PATH,
    os.path.join(OUTPUT_DIR, 'final.subtree'),
    os.path.join(OUTPUT_DIR, 'final.subtree_lambda_ss'),
    os.path.join(OUTPUT_DIR, 'final.subtree_Elogpi'),
    os.path.join(OUTPUT_DIR, 'final.subtree_logEpi'),
    os.path.join(OUTPUT_DIR, 'subgraphs.json'))

print 'Linking visualization code to output directory...'
for basename in ('graph.html', 'subgraphs.html', 'd3.v3.js'):
    os.symlink(os.path.abspath(os.path.join(POSTPROC_DIR, basename)),
        os.path.join(OUTPUT_DIR, basename))

print 'Done:'
print OUTPUT_DIR
