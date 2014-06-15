#!/bin/bash

if [ $# -ne 1 ]
then
    echo 'Specify the /wikipedia/en_id.' >&2
    exit 1
fi

wget "http://en.wikipedia.org/wiki/index.html?curid=$1"
