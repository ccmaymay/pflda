import os
import codecs
from pylowl.proj.brightside.utils import mkdirp_parent


def load_word_set(filename):
    word_set = set()
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            word_set.add(line.strip())
    return word_set


def write_vocab(output_filename, vocab):
    mkdirp_parent(output_filename)
    with codecs.open(output_filename, mode='w', encoding='utf-8') as out_f:
        for (word, word_id) in vocab.items():
            out_f.write(u'%d %s\n' % (word_id, word))
