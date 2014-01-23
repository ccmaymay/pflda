#!/usr/bin/env python


import pylowl # TODO cimport?
import random
import util
import sys
import numpy
cimport numpy
from cpython.exc cimport PyErr_CheckSignals


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

    cdef eval_pl(self, list sample):
        cdef numpy.double_t local_d_count, ll, s, p
        cdef numpy.double_t[:] local_dt_counts, x
        cdef numpy.uint_t i, j, t, w, num_words

        num_words = 0
        ll = 0.0
        local_d_count = 0.0
        local_dt_counts = numpy.zeros((self.num_topics,), dtype=numpy.double)
        x = numpy.zeros((self.num_topics,), dtype=numpy.double)
        for i in xrange(len(sample)):
            local_dt_counts[:] = 0.0
            for j in xrange(len(sample[i])):
                w = sample[i][j]
                for t in xrange(self.num_topics):
                    x[t] = (self.tw_counts[t, w] + self.beta) / (self.t_counts[t] + self.vocab_size * self.beta) * (local_dt_counts[t] + self.alpha) / (local_d_count + self.num_topics * self.alpha)
                s = numpy.sum(x)
                ll += numpy.log(s)
                for t in xrange(self.num_topics):
                    p = x[t] / s
                    local_dt_counts[t] += p
                    local_d_count += p
                num_words += 1
            if i % 1000 == 0:
                PyErr_CheckSignals()

        print('Log-likelihood: %f' % ll)
        print('Perplexity:     %f' % (-ll / num_words))

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
        self.tw_counts[:] = 0
        self.t_counts[:] = 0
        self.dt_counts[:] = 0
        self.d_counts[:] = 0

        for i in xrange(len(sample)):
            for j in xrange(len(sample[i])):
                w = sample[i][j]
                z = random.randint(0, self.num_topics - 1)
                assignments.append(z)
                self.tw_counts[z, w] += 1
                self.t_counts[z] += 1
                self.dt_counts[i, z] += 1
                self.d_counts[i] += 1
            if i % 1000 == 0:
                PyErr_CheckSignals()

        for t in xrange(num_iters):
            m = 0
            for i in xrange(len(sample)):
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
                if i % 1000 == 0:
                    PyErr_CheckSignals()
            sys.stdout.write('.')
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

    def to_string(self, vocab, num_words_per_topic):
        s = ''
        for t in range(self.num_topics):
            s += 'TOPIC %d:\n' % t
            pp = [(word, self.tw_counts[t, w]) for (word, w) in vocab.items()]
            pp.sort(key=lambda p: p[1], reverse=True)
            i = 0
            for (word, count) in pp:
                if count > 0:
                    s += '\t%s (%d)\n' % (word, count)
                i += 1
                if i >= num_words_per_topic:
                    break
        return s


def run_lda(data_dir, *categories):
    reservoir_size = 1000
    num_iters = 1000
    num_topics = 20
    alpha = 0.1
    beta = 0.1

    reservoir = pylowl.ValuedReservoirSampler(reservoir_size)
    dataset = util.Dataset(data_dir, set(categories))
    model = LdaModel(alpha, beta, reservoir_size, num_topics, len(dataset.vocab))

    def preprocess(doc_triple):
        return [dataset.vocab[w] for w in doc_triple[2]]

    test_sample = list(preprocess(t) for t in dataset.test_iterator())

    i = 0
    for doc_triple in dataset.train_iterator():
        if i >= reservoir_size and i % 100 == 0:
            model.learn(reservoir.sample(), num_iters)
            print(model.to_string(dataset.vocab, 20))
            model.eval_pl(test_sample)
        reservoir.insert(doc_triple, preprocess)
        i += 1


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]](*sys.argv[2:])
