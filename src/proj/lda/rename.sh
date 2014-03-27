#!/bin/bash

if [ $# -ne 2 ]
then
    echo 'Specify old and new filename.' >&2
    exit 1
fi

old_fn="$1"
new_fn="$2"

patt=`echo "s#$old_fn#$new_fn#g" | sed 's#\.#\\\.#g'`

git mv "$old_fn" "$new_fn"
sed -i "$patt" *.sh *.qsub experiments/*.sh
