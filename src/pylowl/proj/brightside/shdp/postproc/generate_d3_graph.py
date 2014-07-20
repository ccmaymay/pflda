#!/usr/bin/env python


import os
import json
from pylowl.proj.brightside.corpus import load_vocab
from pylowl.proj.brightside.utils import load_options


DEFAULT_WORDS_PER_TOPIC = 10


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(words_per_topic=DEFAULT_WORDS_PER_TOPIC)
    parser.add_argument('result_dir', type=str,
                        help='path to dir where model was saved')
    parser.add_argument('output_path', type=str,
                        help='output file path')
    parser.add_argument('--words_per_topic', type=int,
                        help='number of words to output per topic')

    args = parser.parse_args()
    generate_d3_graph(
        args.result_dir,
        args.output_path,
        words_per_topic=args.words_per_topic,
    )


def generate_d3_graph(result_dir, output_filename,
                      words_per_topic=DEFAULT_WORDS_PER_TOPIC):
    options = load_options(os.path.join(result_dir, 'options'))
    K = int(options['K'])
    vocab_filename = options['vocab_path']
    lambda_ss_filename = os.path.join(result_dir, 'final.lambda_ss')
    Elogpi_filename = os.path.join(result_dir, 'final.Elogpi')
    logEpi_filename = os.path.join(result_dir, 'final.logEpi')
    Elogtheta_filename = os.path.join(result_dir, 'final.Elogtheta')
    logEtheta_filename = os.path.join(result_dir, 'final.logEtheta')

    vocab = load_vocab(vocab_filename)

    node_topics = []

    for idx in xrange(K):
        node_topics.append({
            'children': [],
            'words': [{'word': vocab[t]} for t in range(len(vocab))]
        })

    for (stat_name, stat_filename) in (('Elogtheta', Elogtheta_filename),
                                       ('logEtheta', logEtheta_filename),
                                       ('lambda_ss', lambda_ss_filename)):
        with open(stat_filename) as f:
            for (idx, line) in enumerate(f):
                for (t, w) in enumerate(float(w) for w in line.strip().split()):
                    node_topics[idx]['words'][t][stat_name] = w
    for node_dict in node_topics:
        node_dict['lambda_ss_sum'] = sum(d['lambda_ss'] for d in node_dict['words'])
        node_dict['words'].sort(key=lambda d: d['lambda_ss'], reverse=True)
        node_dict['words'] = node_dict['words'][:words_per_topic]

    for (stat_name, stat_filename) in (('Elogpi', Elogpi_filename),
                                       ('logEpi', logEpi_filename)):
        with open(stat_filename) as f:
            for (idx, line) in enumerate(f):
                w = float(line.strip())
                node_topics[idx][stat_name] = w

    for idx in xrange(1, K):
        parent_node_dict = node_topics[idx-1]
        node_dict = node_topics[idx]
        parent_node_dict['children'].append(node_dict)

    with open(output_filename, 'w') as f:
        json.dump(node_topics[0], f, indent=2)


if __name__ == '__main__':
    main()
