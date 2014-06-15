#!/usr/bin/env python


import utils
import subprocess
import tempfile
import os
import math


NUM_TYPES_PER_TOPIC = 10
MIN_FONTSIZE = 8
MAX_FONTSIZE = 18
MAX_NODE_WIDTH = 30
GRAPHVIZ_CMD = 'twopi' # 'dot'


def transform_type(t, weight, topic_weight):
    weight_frac = weight/topic_weight
    if weight_frac > 0.3:
        return '***%s***' % t
    elif weight_frac > 0.1:
        return '**%s**' % t
    elif weight_frac > 0.03:
        return '*%s*' % t
    else:
        return t


def fontsize(topic_weight, max_weight):
    return max(MIN_FONTSIZE, MAX_FONTSIZE + math.log(topic_weight/max_weight))


def main(trunc_csv, input_filename, output_filename):
    trunc = [int(t) for t in trunc_csv.split(',')]

    (gv_output_fd, gv_output_filename) = tempfile.mkstemp('generate_topic_graphs.XXXXXX')
    with open(input_filename) as f:
        max_weight = max(sum(float(w) for w in line.strip().split()[1::2]) for line in f)

    with os.fdopen(gv_output_fd, 'w') as out_f:
        out_f.write('graph {\n')
        out_f.write('  graph [overlap=false, root=0];\n\n')
        with open(input_filename) as f:
            n = 0
            for line in f:
                label = []
                label_line = []
                label_line_len = 0
                pieces = line.strip().split()
                topic_weight = sum(float(w) for w in pieces[1::2])
                types = pieces[0:(2*NUM_TYPES_PER_TOPIC):2]
                weights = [float(w) for w in pieces[1:(2*NUM_TYPES_PER_TOPIC):2]]
                for (t, weight) in zip(types, weights):
                    t = transform_type(t, weight, topic_weight)

                    if not label_line:
                        label_line.append(t)
                        label_line_len += len(t)
                    else:
                        if label_line_len + len(t) + len(label_line) > MAX_NODE_WIDTH:
                            label.append(label_line)
                            label_line = [t]
                            label_line_len = len(t)
                        else:
                            label_line.append(t)
                            label_line_len += len(t)
                if label_line:
                    label.append(label_line)
                out_f.write('  %d[label="%s", fontsize=%f];\n' % (n, r'\n'.join(' '.join(ll) for ll in label), fontsize(topic_weight, max_weight)))
                n += 1

        out_f.write('\n')

        m = utils.tree_index_m(trunc)
        b = utils.tree_index_b(trunc)
        for node in utils.tree_iter(trunc):
            idx = utils.tree_index(node, m, b)
            if len(node) < len(trunc): # inner node
                for j in xrange(trunc[len(node)]):
                    child = node + (j,)
                    c_idx = utils.tree_index(child, m, b)
                    out_f.write('  %d -- %d;\n' % (idx, c_idx))

        out_f.write('}\n')

    subprocess.call([GRAPHVIZ_CMD, '-Tsvg', '-o%s' % output_filename, gv_output_filename])
    os.remove(gv_output_filename)


if __name__ == '__main__':
    import sys

    args = []
    params = dict()
    for token in sys.argv[1:]:
        eq_pos = token.find('=')
        if token.startswith('--') and eq_pos >= 0:
            k = token[len('--'):eq_pos]
            v = token[(eq_pos+1):len(token)]
            params[k] = v
        else:
            args.append(token)

    main(*args, **params)
