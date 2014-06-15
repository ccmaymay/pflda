#!/bin/bash

f=`mktemp`
g=`mktemp`
cut -f 3-4 data/cities.inc_dc.tab | sed 's#	#/#' | sort | uniq | sed 's/ /_/g' > "$f"
sort data/cities.inc_dc.txt > "$g"
diff "$f" "$g" | grep '>' | sed 's/^> //' > data/missing_cities.txt
rm -f "$f"
rm -f "$g"
cat data/missing_cities.txt
