#!/bin/bash
stats_file=`mktemp XXXXXX.stats`
echo "$stats_file"
LD_LIBRARY_PATH=..:. PYTHONPATH=..:. python -c "import cProfile; cProfile.run('import run_lda', '$stats_file')"
