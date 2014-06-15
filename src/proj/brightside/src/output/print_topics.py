#!/usr/bin/env python


def main(topics_filename, vocab_filename, num_words=None):
    if num_words is None:
        num_words = 20
    else:
        num_words = int(num_words)

    vocab = dict()

    with open(vocab_filename) as f:
        for line in f:
            pieces = line.split()
            word_id = int(pieces[0])
            word = pieces[1]
            vocab[word] = word_id

    rvocab = [None for i in range(len(vocab))]
    for (word, word_id) in vocab.items():
        rvocab[word_id] = word

    with open(topics_filename) as f:
        for line in f:
            pieces = line.strip().split()
            p = [(i, float(pieces[i])) for i in range(len(pieces))]
            p.sort(key=lambda x: x[1], reverse=True)
            print ' '.join((rvocab[x[0]] for x in p[:num_words]))


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
