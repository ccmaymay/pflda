#!/bin/bash
stats_file=`mktemp XXXXXX.stats`
echo "$stats_file"
python -c "import cProfile; cProfile.run('import run_lda', '$stats_file')"
