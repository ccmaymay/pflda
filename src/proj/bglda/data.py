#!/usr/bin/env python


import os
import random
import gzip


OOV = '_OOV_'
TOKEN_START = 2


def _sorted_file_paths(dir_path, ext=None):
    paths = []
    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and not filename.startswith('.') and (ext is None or filename.endswith(ext)):
            paths.append(path)
    paths.sort()
    return paths


def _increment(table, key):
    if key in table:
        table[key] += 1
    else:
        table[key] = 1


class Dataset(object):
    def __init__(self, data_dir, shuffle=False, oov_max_count=None,
            test_in_vocab=False):
        if oov_max_count is None:
            oov_max_count = 1

        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.shuffle = shuffle

        word_counts = dict()
        self._increment_word_counts(word_counts, self.train_dir)
        if test_in_vocab:
            self._increment_word_counts(word_counts, self.test_dir)

        self.vocab = {OOV: 0}
        for (word, count) in word_counts.items():
            if count > oov_max_count:
                self.vocab[word] = len(self.vocab)

    def _increment_word_counts(self, word_counts, split_dir):
        for path in _sorted_file_paths(split_dir, '.gz'):
            with gzip.open(path) as f:
                for line in f:
                    words = self._parse_doc(line)
                    for word in words:
                        _increment(word_counts, word)

    def _parse_doc(self, line):
        return line.split()[TOKEN_START:]

    def _replace_oov(self, word):
        if word in self.vocab:
            return word
        else:
            return OOV

    def _iterator(self, dir_path):
        if self.shuffle:
            items = list(self._raw_iterator(dir_path))
            random.shuffle(items)
            for item in items:
                yield item
        else:
            for item in self._raw_iterator(dir_path):
                yield item

    def _raw_iterator(self, dir_path):
        for path in _sorted_file_paths(dir_path, '.gz'):
            with gzip.open(path) as f:
                for line in f:
                    words = self._parse_doc(line)
                    yield [self._replace_oov(w) for w in words]

    def train_iterator(self):
        return self._iterator(self.train_dir)

    def test_iterator(self):
        return self._iterator(self.test_dir)
