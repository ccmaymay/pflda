#!/usr/bin/env python


import os
import random
import re
import gzip


WHITESPACE_RE = re.compile(r'\s+')
NON_ALPHA_RE = re.compile(r'[^a-zA-Z]+')
OOV = '_OOV_'


def _sorted_file_paths(dir_path, ext=None):
    paths = []
    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and not filename.startswith('.') and (ext is None or filename.endswith(ext)):
            paths.append(path)
    paths.sort()
    return paths


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


class CompoundTokenFilter(object):
    def __init__(self):
        self.filters = []

    def add(self, f):
        self.filters.append(f)

    def filter(self, words):
        filtered = words
        for f in self.filters:
            filtered = f.filter(filtered)
        return filtered


class BlacklistTokenFilter(object):
    def __init__(self, blacklist=None):
        if blacklist is None:
            self.bl_set = set()
        else:
            self.bl_set = set(blacklist)

    def filter(self, words):
        return [w for w in words if w not in self.bl_set]


class NonEmptyTokenFilter(object):
    def filter(self, words):
        return [w for w in words if w]


class LowerTokenFilter(object):
    def filter(self, words):
        return [w.lower() for w in words]


class WhitespaceTokenizer(object):
    def tokenize(self, s):
        return [w for w in WHITESPACE_RE.split(s) if NON_ALPHA_RE.search(w) is None]


class NonAlphaTokenizer(object):
    def tokenize(self, s):
        return [w for w in NON_ALPHA_RE.split(s)]


class TopicList(object):
    def __init__(self, filename):
        self.topics = list()
        with open(filename) as f:
            for line in f:
                if line.strip():
                    tokens_and_counts = line.strip().split()
                    if len(tokens_and_counts) % 2 != 0:
                        raise Exception('Topic distribution is malformatted: expected even-length list.')
                    topic = dict()
                    for i in xrange(len(tokens_and_counts)/2):
                        token = tokens_and_counts[2*i]
                        count = tokens_and_counts[2*i + 1]
                        topic[token] = int(count)
                    self.topics.append(topic)

    def topic(self, i):
        return self.topics[i]

    def num_topics(self):
        return len(self.topics)


class Dataset(object):
    def __init__(self, data_dir, categories, shuffle=False):
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.categories = categories
        self.shuffle = shuffle

        word_counts = dict()

        for path in _sorted_file_paths(self.train_dir, '.gz'):
            with gzip.open(path) as f:
                for line in f:
                    (doc_idx, category, words) = self._parse_doc(line)
                    if category in categories:
                        for word in words:
                            _increment(word_counts, word)

        self.vocab = {OOV: 0}
        for (word, count) in word_counts.items():
            if count > 1:
                self.vocab[word] = len(self.vocab)

    def _parse_doc(self, line):
        tokens = line.split()
        doc_idx = int(tokens[0])
        category = tokens[1]
        words = tokens[2:]
        return (doc_idx, category, words)

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
        self.f.write('%d %s %s\n' % (doc_idx, category, ' '.join(tokens)))

    def close(self):
        self.f.close()


def _parse_comm(comm):
    tokens = []
    if comm.sectionSegmentations is not None:
        for sectionSegmentation in comm.sectionSegmentations:
            if sectionSegmentation.sectionList is not None:
                for section in sectionSegmentation.sectionList:
                    if section.sentenceSegmentation is not None:
                        for sentenceSegmentation in section.sentenceSegmentation:
                            if sentenceSegmentation.sentenceList is not None:
                                for sentence in sentenceSegmentation.sentenceList:
                                    if sentence.tokenizationList is not None:
                                        for tokenization in sentence.tokenizationList:
                                            if tokenization.tokenList is not None:
                                                for token in tokenization.tokenList:
                                                    tokens.append(token.text)
    return tokens


def transform_tng(train_input_dir, test_input_dir, base_output_dir, split_mode=None, stop_list_path=None, remove_header=False, remove_walls=False, lower=False):
    train_output_dir = os.path.join(base_output_dir, 'train')
    test_output_dir = os.path.join(base_output_dir, 'test')

    token_filter = CompoundTokenFilter()
    token_filter.add(NonEmptyTokenFilter())
    if lower:
        token_filter.add(LowerTokenFilter())
    if stop_list_path is not None:
        token_filter.add(BlacklistTokenFilter(_load_stop_set(stop_list_path)))

    if split_mode is None or split_mode == 'nonalpha':
        tokenizer = NonAlphaTokenizer()
    elif split_mode == 'whitespace':
        tokenizer = WhitespaceTokenizer()
    else:
        raise Exception('Unknown split mode %s' % split_mode)

    doc_idx = 0

    for (input_dir, output_dir) in ((train_input_dir, train_output_dir), (test_input_dir, test_output_dir)):
        writer = DatasetWriter(os.path.join(output_dir, 'all.gz'))
        for (category, path) in _tng_shuffled_nested_file_paths(input_dir):
            with open(path) as f:
                seen_empty_line = False
                tokens = []
                for line in f:
                    line = line.rstrip()
                    if line:
                        contains_ws = WHITESPACE_RE.search(line) is not None
                        if (seen_empty_line or not remove_header) and (contains_ws or not remove_walls):
                            tokens.extend(token_filter.filter(tokenizer.tokenize(line)))
                    else:
                        seen_empty_line = True
                writer.write(doc_idx, category, tokens)
            doc_idx += 1
        writer.close()


def transform_twitter(input_path, base_output_dir, train_frac=None, stop_list_path=None):
    train_output_dir = os.path.join(base_output_dir, 'train')
    test_output_dir = os.path.join(base_output_dir, 'test')

    if train_frac is None:
        train_frac = 0.6
    else:
        train_frac = float(train_frac)

    token_filter = CompoundTokenFilter()
    token_filter.add(NonEmptyTokenFilter())
    if stop_list_path is not None:
        token_filter.add(BlacklistTokenFilter(_load_stop_set(stop_list_path)))

    tokenizer = WhitespaceTokenizer()

    train_writer = DatasetWriter(os.path.join(train_output_dir, 'all.gz'))
    test_writer = DatasetWriter(os.path.join(test_output_dir, 'all.gz'))

    category = 'null'
    doc_idx = 0
    with open(input_path) as f:
        for line in f:
            first_split = line.find(' ') + 1
            tokens = token_filter.filter(tokenizer.tokenize(line[first_split:]))
            if random.random() < train_frac:
                train_writer.write(doc_idx, category, tokens)
            else:
                test_writer.write(doc_idx, category, tokens)
            doc_idx += 1

    train_writer.close()
    test_writer.close()


def transform_gigaword(input_dir, base_output_dir, train_frac=None, split_mode=None, stop_list_path=None, lower=False):
    if train_frac is None:
        train_frac = 0.8
    else:
        train_frac = float(train_frac)

    train_output_dir = os.path.join(base_output_dir, 'train')
    test_output_dir = os.path.join(base_output_dir, 'test')

    token_filter = CompoundTokenFilter()
    token_filter.add(NonEmptyTokenFilter())
    if lower:
        token_filter.add(LowerTokenFilter())
    if stop_list_path is not None:
        token_filter.add(BlacklistTokenFilter(_load_stop_set(stop_list_path)))

    if split_mode is None or split_mode == 'nonalpha':
        tokenizer = NonAlphaTokenizer()
    elif split_mode == 'whitespace':
        tokenizer = WhitespaceTokenizer()
    else:
        raise Exception('Unknown split mode %s' % split_mode)

    category = 'null'
    doc_idx = 0

    train_writer = DatasetWriter(os.path.join(train_output_dir, 'all.gz'))
    test_writer = DatasetWriter(os.path.join(test_output_dir, 'all.gz'))

    for path in _sorted_file_paths(input_dir, '.gz'):
        with gzip.open(path) as f:
            in_doc = False
            in_p = False
            for line in f:
                line = line.rstrip()
                if line:
                    if line.lower().startswith('<doc'):
                        in_doc = True
                        tokens = []
                    elif line.lower().startswith('</doc'):
                        in_doc = False
                        if tokens:
                            if random.random() < train_frac:
                                train_writer.write(doc_idx, category, tokens)
                            else:
                                test_writer.write(doc_idx, category, tokens)
                            doc_idx += 1
                    elif line.lower().startswith('<p'):
                        in_p = True
                    elif line.lower().startswith('</p'):
                        in_p = False
                    elif in_doc and in_p:
                        tokens.extend(token_filter.filter(tokenizer.tokenize(line)))

    train_writer.close()
    test_writer.close()


def transform_concrete(input_dir, base_output_dir, train_frac=None, stop_list_path=None):
    from thrift.transport import TSocket, TTransport
    from thrift.protocol import TBinaryProtocol
    from concrete.communication.ttypes import Communication

    if train_frac is None:
        train_frac = 0.8
    else:
        train_frac = float(train_frac)

    train_output_dir = os.path.join(base_output_dir, 'train')
    test_output_dir = os.path.join(base_output_dir, 'test')

    token_filter = CompoundTokenFilter()
    token_filter.add(NonEmptyTokenFilter())
    if stop_list_path is not None:
        token_filter.add(BlacklistTokenFilter(_load_stop_set(stop_list_path)))

    category = 'null'
    doc_idx = 0

    train_writer = DatasetWriter(os.path.join(train_output_dir, 'all.gz'))
    test_writer = DatasetWriter(os.path.join(test_output_dir, 'all.gz'))

    for path in _sorted_file_paths(input_dir):
        with open(path, 'rb') as f:
            transportIn = TTransport.TFileObjectTransport(f)
            protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
            comm = Communication()
            comm.read(protocolIn)
            tokens = _parse_comm(comm)
            if tokens:
                if random.random() < train_frac:
                    train_writer.write(doc_idx, category, tokens)
                else:
                    test_writer.write(doc_idx, category, tokens)
                doc_idx += 1

    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    import sys

    f = globals()[sys.argv[1]]

    args = []
    kwargs = dict()
    for (i, token) in zip(range(2, len(sys.argv)), sys.argv[2:]):
        eq_pos = token.find('=')
        if token.startswith('--') and eq_pos >= 0:
            k = token[len('--'):eq_pos]
            v = token[(eq_pos+1):len(token)]
            kwargs[k] = v
        else:
            args.append(token)

    f(*args, **kwargs)
