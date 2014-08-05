#!/usr/bin/env python

import codecs
import json
import random


def shuffle_json(input_path, output_path):
    with codecs.open(input_path, encoding='utf-8') as f:
        json_list = json.load(f)
        random.shuffle(json_list)
    with codecs.open(output_path, encoding='utf-8', mode='w') as f:
        json.dump(json_list, f, indent=2)


if __name__ == '__main__':
    import sys
    shuffle_json(*sys.argv[1:])
