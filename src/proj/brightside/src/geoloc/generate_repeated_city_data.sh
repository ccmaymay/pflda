#!/bin/bash

while read line
do
    s=`echo "$line" | cut -c -2`
    c=`echo "$line" | cut -c 4- | sed 's/ /_/g'`
    grep "	$s	$c	" data/cities.inc_dc.tab
done < data/repeated_cities.txt > data/repeated_cities.inc_dc.tab
