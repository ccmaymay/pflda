#!/usr/bin/env python


import pylowl # TODO cimport?
import random
import util
import sys
import numpy
cimport numpy


cdef class LdaModel(object):
    cdef numpy.double_t alpha, beta
    cdef numpy.uint_t num_topics, num_docs, vocab_size
    cdef numpy.uint_t[:, ::1] tw_counts, dt_counts
    cdef numpy.uint_t[:] t_counts, d_counts
    cdef numpy.double_t[:] pmf

    def __cinit__(self, numpy.double_t alpha, numpy.double_t beta, numpy.uint_t num_docs, numpy.uint_t num_topics, numpy.uint_t vocab_size):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.num_docs = num_docs
        self.vocab_size = vocab_size
        self.tw_counts = numpy.zeros((num_topics, vocab_size), dtype=numpy.uint)
        self.t_counts = numpy.zeros((num_topics,), dtype=numpy.uint)
        self.dt_counts = numpy.zeros((num_docs, num_topics), dtype=numpy.uint)
        self.d_counts = numpy.zeros((num_docs,), dtype=numpy.uint)
        self.pmf = numpy.zeros((num_topics,), dtype=numpy.double)

    cdef numpy.double_t conditional_posterior(self, numpy.uint_t i, numpy.uint_t w, numpy.uint_t t):
        return (self.tw_counts[t, w] + self.beta) / (self.t_counts[t] + self.vocab_size * self.beta) * (self.dt_counts[i, t] + self.alpha) / (self.d_counts[i] + self.num_topics * self.alpha)

    cdef numpy.uint_t sample_topic(self, numpy.uint_t i, numpy.uint_t w):
        cdef numpy.double_t prior, r
        cdef numpy.uint_t t

        prior = 0.0
        for t in xrange(self.num_topics):
            self.pmf[t] = self.conditional_posterior(i, w, t)
            prior += self.pmf[t]

        r = random.random() * prior
        for t in xrange(self.num_topics-1):
            if r < self.pmf[t]:
                return t
            self.pmf[t+1] += self.pmf[t]

        return self.num_topics - 1

    cdef learn(self, list sample, numpy.uint_t num_iters):
        cdef list assignments
        cdef numpy.uint_t t, i, j, w, m, z

        assignments = []
        # TODO
        #self.tw_counts = [0] * (self.num_topics * self.vocab_size)
        #self.t_counts = [0] * self.num_topics
        #self.dt_counts = [0] * (self.num_docs * self.num_topics)
        #self.d_counts = [0] * self.num_docs

        for i in xrange(self.num_docs):
            for j in xrange(len(sample[i])):
                w = sample[i][j]
                z = random.randint(0, self.num_topics - 1)
                assignments.append(z)
                self.tw_counts[z, w] += 1
                self.t_counts[z] += 1
                self.dt_counts[i, z] += 1
                self.d_counts[i] += 1

        for t in xrange(num_iters):
            m = 0
            for i in xrange(self.num_docs):
                for j in xrange(len(sample[i])):
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

    def to_string(self, vocab):
        s = ''
        for t in range(self.num_topics):
            s += 'TOPIC %d:\n' % t
            pp = [(word, self.tw_counts[t, w]) for (word, w) in vocab.items()]
            pp.sort(key=lambda p: p[1], reverse=True)
            for (word, count) in pp:
                if count > 0:
                    s += '\t%s (%d)\n' % (word, count)
        return s


def run_lda(data_dir, *categories):
    reservoir_size = 1000
    num_iters = 1000
    num_topics = 3
    alpha = 0.1
    beta = 0.1

    reservoir = pylowl.ValuedReservoirSampler(reservoir_size)
    dataset = util.Dataset(data_dir, set(categories))
    model = LdaModel(alpha, beta, reservoir_size, num_topics, len(dataset.vocab))

    def preprocess(doc_triple):
        return [dataset.vocab[w] for w in doc_triple[2]]

    i = 0
    for doc_triple in dataset.train_iterator():
        if i >= reservoir_size and i % 100 == 0:
            model.learn(reservoir.sample(), num_iters)
            print(model.to_string(dataset.vocab))
        reservoir.insert(doc_triple, preprocess)
        i += 1


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]](*sys.argv[2:])
