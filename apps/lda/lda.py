#!/usr/bin/env python


import pylowl
import random
import re


OOV = '_OOV_'
NON_ALPHA = re.compile(r'[^a-zA-Z]')


class LdaModel(object):
    def __init__(self, reservoir_size, alpha, beta, num_topics, vocab):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.vocab = vocab
        self.r_vocab = [None] * vocab.size
        for (word, idx) in vocab.items():
            self.r_vocab[idx] = word
        self.tw_counts = None
        self.t_counts = None
        self.dt_counts = None
        self.d_counts = None
        self.reservoir = pylowl.ValuedReservoirSampler(reservoir_size)

    def conditional_posterior(self, i, w, t):
        return (self.tw_counts[t * self.vocab.size + w] + self.beta) / float(self.t_counts[t] + self.vocab.size * self.beta) * (self.dt_counts[i * self.num_topics + t] + self.alpha) / float(self.d_counts[i] + self.num_topics * self.alpha)

    def tokenize(self, doc):
        return [self.vocab.get(word, self.vocab[OOV]) for word in doc.strip().split()]

    def sample_topic(self, i, w):
        pmf = [self.conditional_posterior(i, w, t) for t in range(self.num_topics)]
        cmf = self.make_cmf(pmf)

        r = random.random()
        for k in range(len(cmf)):
            if r < cmf[k]:
                return k
        return len(cmf) - 1

    def make_cmf(self, pmf):
        cmf = [pmf[0]]
        for p in pmf[1:]:
            cmf.append(cmf[-1] + p)
        cmf[-1] = 1
        return cmf

    def add_doc(self, line):
        self.reservoir.insert(line, tokenize)

    def learn(self, num_iters):
        sample = self.reservoir.sample()
        assignments = []
        self.tw_counts = [0] * (self.num_topics * self.vocab.size)
        self.t_counts = [0] * self.num_topics
        self.dt_counts = [0] * (len(sample) * self.num_topics)
        self.d_counts = [0] * len(sample)
        for i in range(len(sample)):
            for j in sample[i]:
                w = sample[i][j]
                assignments.append(random.randint(0, self.num_topics - 1))
                self.tw_counts[assignments[-1] * self.vocab.size + w] += 1
                self.t_counts[assignments[-1]] += 1
                self.dt_counts[i * self.num_topics + assignments[-1]] += 1
                self.d_counts[i] += 1
        for t in range(num_iters):
            m = 0
            for i in range(len(sample)):
                for j in sample[i]:
                    w = sample[i][j]
                    self.tw_counts[assignments[m] * self.vocab.size + w] -= 1
                    self.t_counts[assignments[m]] -= 1
                    self.dt_counts[i * self.num_topics + assignments[m]] -= 1
                    self.d_counts[i] -= 1
                    assignments[m] = self.sample_topic(i, w)
                    self.tw_counts[assignments[m] * self.vocab.size + w] += 1
                    self.t_counts[assignments[m]] += 1
                    self.dt_counts[i * self.num_topics + assignments[m]] += 1
                    self.d_counts[i] += 1
                    m += 1

    def __str__(self):
        s = ''
        for t in range(self.num_topics):
            s += 'TOPIC %d:\n' % t
            pp = [(word, self.tw_counts[t * self.vocab_size + self.vocab[word]]) for word in self.vocab]
            pp.sort(key=lambda p: p[1], reverse=True)
            for (word, count) in pp:
                s += '\t%s (%d)\n' % (word, count)
        return s


def tokenize(line):
    return (w.lower() for w in NON_ALPHA.split(line.strip()) if w)
            
    
def run_lda(vocab_filename, docs_filename):
    vocab = {OOV: 0}
    with open(vocab_filename) as f:
        for line in f:
            word = line.strip()
            vocab[word] = len(vocab)
    model = LdaModel(1000, 0.1, 0.1, 10, vocab)
    with open(docs_filename) as f:
        for line in f:
            model.add_doc(line)
            model.learn(1000)
            print(model)


def make_vocab(vocab_filename, docs_filename):
    word_counts = dict()
    with open(docs_filename) as f:
        for line in f:
            for token in tokenize(line):
                if token in word_counts:
                    word_counts[token] += 1
                else:
                    word_counts[token] = 1
    with open(vocab_filename, 'w') as f:
        for (word, count) in word_counts.items():
            if count > 1:
                f.write(word + '\n')


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]](*sys.argv[2:])
