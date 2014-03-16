#!/bin/bash
LD_LIBRARY_PATH=..:.:$LD_LIBRARY_PATH PYTHONPATH=..:.:$PYTHONPATH python run_gibbs.py $@
