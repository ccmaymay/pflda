#!/bin/bash
for h in `qstat | awk '{ print $8 }' | grep text.q | cut -d. -f 2 | cut -c 3- | sort | uniq`
do
    ssh $h ps -o rss,vsz,cmd | grep run_pf
done
