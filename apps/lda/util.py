#!/usr/bin/env python


import os
import random
import re
import gzip


NON_ALPHA_RE = re.compile(r'[^a-zA-Z]+')
OOV = '_OOV_'


def _load_stop_set(filename):
    stop_set = set()
    with open(filename) as f:
        for line in f:
            stripped_line = line.rstrip()
            if stripped_line:
                stop_set.add(stripped_line)
    return stop_set


def _increment(table, key):
    if key in table:
        table[key] += 1
    else:
        table[key] = 1


class Tokenizer(object):
    def __init__(self, stop_set=None):
        if stop_set is None:
            self.stop_set = set()
        else:
            self.stop_set = stop_set

    def tokenize(s):
        return [w for w in s.lower().split(NON_ALPHA_RE) if w and w not in stop_set]


class Dataset(object):
    def __init__(self, data_dir, categories):
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.categories = categories

        word_counts = dict()

        for path in self._sorted_file_paths(self.train_dir):
            with gzip.open(path) as f:
                for line in f:
                    (doc_idx, category, words) = self._parse_doc(line)
                    if category in categories:
                        for word in words:
                            _increment(word_counts, word)

        self.vocab = set(k for (k, v) in word_counts.items() if v > 1)
        self.vocab.add(OOV)

    def _parse_doc(self, line):
        tokens = line.split()
        doc_idx = int(tokens[0])
        category = tokens[1]
        words = tokens[2:]
        return (doc_idx, category, words)

    def _sorted_file_paths(self, dir_path):
        paths = []
        for filename in os.listdir(dir_path):
            path = os.path.join(dir_path, filename)
            if os.path.isfile(path) and not filename.startswith('.') and filename.endswith('.gz'):
                paths.append(path)
        paths.sort()
        return paths

    def _replace_oov(self, word):
        if word in self.vocab:
            return word
        else:
            return OOV

    def _iterator(self, dir_path):
        for path in self._sorted_file_paths(dir_path):
            with gzip.open(path) as f:
                for line in f:
                    (doc_idx, category, words) = self._parse_doc(line)
                    if category in self.categories:
                        yield (doc_idx, category, [self._replace_oov(w) for w in words])

    def train_iterator(self):
        return self._iterator(self.train_dir)

    def test_iterator(self):
        return self._iterator(self.test_dir)


def _tng_shuffled_nested_file_paths(input_dir):
    category_path_pairs = []
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not category.startswith('.') and os.path.isdir(category_path):
            for doc in os.listdir(category_path):
                doc_path = os.path.join(category_path, doc)
                if not doc.startswith('.') and os.path.isfile(doc_path):
                    category_path_pairs.append((category, doc_path))
    random.shuffle(category_path_pairs)
    return category_path_pairs


class DatasetWriter(object):
    def __init__(self, filename):
        os.makedirs(os.path.dirname(filename))
        self.f = gzip.open(filename, 'w')

    def write(self, doc_idx, category, tokens):
        f.write('%d %s %s\n' % (doc_idx, category, ' '.join(tokens)))

    def close(self):
        f.close()


def transform_tng(train_input_dir, test_input_dir, base_output_dir):
    train_output_dir = os.path.join(base_output_dir, 'train')
    test_output_dir = os.path.join(base_output_dir, 'test')

    tokenizer = Tokenizer(STOP_LIST_PATH)
    doc_idx = 0

    for ((input_dir, output_dir) in ((train_input_dir, train_output_dir), (test_input_dir, test_output_dir)):
        writer = DatasetWriter(os.path.join(output_dir, 'all.gz'))
        for (category, path) in _tng_shuffled_nested_file_paths(input_dir):
            with open(path) as f:
                tokens = []
                for line in f:
                    tokens.extend(tokenizer.tokenize(line))
                writer.write(doc_idx, category, tokens)
            doc_idx += 1
        writer.close()


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]](*sys.argv[2:])
