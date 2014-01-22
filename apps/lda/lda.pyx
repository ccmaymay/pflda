#!/usr/bin/env python


import sys
import pylowl
import random
import re
import util
import numpy


class LdaModel(object):
    def __init__(self, reservoir_size, alpha, beta, num_topics, vocab):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.vocab = vocab
        self.reservoir = pylowl.ValuedReservoirSampler(reservoir_size)

    def conditional_posterior(self, i, w, t):
        return (self.tw_counts[t, w] + self.beta) / float(self.t_counts[t] + len(self.vocab) * self.beta) * (self.dt_counts[i, t] + self.alpha) / float(self.d_counts[i] + self.num_topics * self.alpha)

    def sample_topic(self, i, w):
        dist = [self.conditional_posterior(i, w, t) for t in range(self.num_topics)]
        r = random.random() * sum(dist)
        for k in range(len(dist)-1):
            if r < dist[k]:
                return k
            dist[k+1] += dist[k]
        return len(dist) - 1

    def preprocess(self, doc_triple):
        return [self.vocab[w] for w in doc_triple[2]]

    def add_doc(self, doc_triple):
        self.reservoir.insert(doc_triple, self.preprocess)

    def learn(self, num_iters):
        sample = self.reservoir.sample()
        assignments = numpy.zeros((sum(len(doc) for doc in sample),), dtype=numpy.int)
        self.tw_counts = numpy.zeros((self.num_topics, len(self.vocab)), dtype=numpy.int)
        self.t_counts = numpy.zeros((self.num_topics,), dtype=numpy.int)
        self.dt_counts = numpy.zeros((len(sample), self.num_topics), dtype=numpy.int)
        self.d_counts = numpy.zeros((len(sample),), dtype=numpy.int)

        m = 0
        for i in range(len(sample)):
            for j in range(len(sample[i])):
                w = sample[i][j]
                z = random.randint(0, self.num_topics - 1)
                assignments[m] = z
                self.tw_counts[z, w] += 1
                self.t_counts[z] += 1
                self.dt_counts[i, z] += 1
                self.d_counts[i] += 1
                m += 1

        for t in range(num_iters):
            m = 0
            for i in range(len(sample)):
                for j in range(len(sample[i])):
                    w = sample[i][j]
                    z = assignments[m]
                    self.tw_counts[z, w] -= 1
                    self.t_counts[z] -= 1
                    self.dt_counts[i, z] -= 1
                    self.d_counts[i] -= 1
                    z = self.sample_topic(i, w)
                    assignments[m] = z
                    self.tw_counts[z, w] += 1
                    self.t_counts[z] += 1
                    self.dt_counts[i, z] += 1
                    self.d_counts[i] += 1
                    m += 1
            sys.stdout.write('.')
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

    def __str__(self):
        s = ''
        for t in range(self.num_topics):
            s += 'TOPIC %d:\n' % t
            pp = [(word, self.tw_counts[t, self.vocab[word]]) for word in self.vocab]
            pp.sort(key=lambda p: p[1], reverse=True)
            for (word, count) in pp:
                if count > 0:
                    s += '\t%s (%d)\n' % (word, count)
        return s


def run_lda(data_dir, *categories):
    dataset = util.Dataset(data_dir, set(categories))
    model = LdaModel(1000, 0.1, 0.1, 3, dataset.vocab)
    i = 0
    for doc_triple in dataset.train_iterator():
        if i > 0 and i % 100 == 0:
            model.learn(1000)
            print(model)
        model.add_doc(doc_triple)
        i += 1


if __name__ == '__main__':
    globals()[sys.argv[1]](*sys.argv[2:])
