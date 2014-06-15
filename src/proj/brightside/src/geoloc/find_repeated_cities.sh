#!/bin/bash

cut -f 3-4 data/cities.inc_dc.tab | sed 's#	#/#' | sort | uniq -d | sed 's/ /_/g' > data/repeated_cities.txt
cat data/repeated_cities.txt
