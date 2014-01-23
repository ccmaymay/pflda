#!/usr/bin/env python


import pylowl
import random
from data import Dataset
import sys
import numpy
cimport numpy
from cpython.exc cimport PyErr_CheckSignals


cdef numpy.double_t entropy1(list labels, list label_types):
    cdef numpy.uint_t i, j
    cdef numpy.double_t n, count, p, _entropy

    _entropy = 0.0
    n = float(len(labels))
    for i in xrange(len(label_types)):
        count = 0.0
        for j in xrange(len(labels)):
            if labels[j] == label_types[i]:
                count += 1.0
        p = count / n
        if p > 0.0:
            _entropy += -p * numpy.log(p)

    return _entropy


cdef numpy.double_t entropy2(long[:] inferred_topics,
        numpy.uint_t num_topics):
    cdef numpy.uint_t i, t
    cdef numpy.double_t n, count, p, _entropy

    _entropy = 0.0
    n = float(inferred_topics.shape[0])
    for t in xrange(num_topics):
        count = 0.0
        for i in xrange(inferred_topics.shape[0]):
            if inferred_topics[i] == t:
                count += 1.0
        p = count / n
        if p > 0.0:
            _entropy += -p * numpy.log(p)

    return _entropy


cdef numpy.double_t mi(list labels, list label_types,
        long[:] inferred_topics, numpy.uint_t num_topics):
    cdef numpy.uint_t i, t, j
    cdef numpy.double_t n, count, marginal_count1, marginal_count2, _mi

    _mi = 0.0
    n = float(len(labels))
    for i in xrange(len(label_types)):
        for t in xrange(num_topics):
            count = 0.0
            marginal_count1 = 0.0
            marginal_count2 = 0.0
            for j in xrange(len(labels)):
                if labels[j] == label_types[i]:
                    marginal_count1 += 1.0
                if inferred_topics[j] == t:
                    marginal_count2 += 1.0
                if labels[j] == label_types[i] and inferred_topics[j] == t:
                    count += 1.0
            if count > 0.0:
                _mi += (count / n) * (numpy.log(count * n) - numpy.log(marginal_count1 * marginal_count2))

    return _mi


cdef numpy.double_t nmi(list labels, list label_types,
        numpy.uint_t[:, ::1] dt_counts, numpy.uint_t num_topics):
    cdef long[:] inferred_topics
    cdef numpy.double_t _nmi

    inferred_topics = numpy.argmax(dt_counts, 1)
    _nmi = 2.0 * mi(labels, label_types, inferred_topics, num_topics) / (entropy1(labels, label_types) + entropy2(inferred_topics, num_topics))

    return _nmi


cdef class GlobalParams:
    cdef numpy.double_t alpha, beta
    cdef numpy.uint_t num_topics, vocab_size
    cdef numpy.uint_t[:, ::1] tw_counts
    cdef numpy.uint_t[:] t_counts

    def __cinit__(self, numpy.double_t alpha, numpy.double_t beta,
            numpy.uint_t num_topics, numpy.uint_t vocab_size):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.tw_counts = numpy.zeros((num_topics, vocab_size), dtype=numpy.uint)
        self.t_counts = numpy.zeros((num_topics,), dtype=numpy.uint)

    cdef void clear(self):
        self.tw_counts[:] = 0
        self.t_counts[:] = 0

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

    cdef GlobalParams copy(self):
        cdef GlobalParams c
        c = GlobalParams(self.alpha, self.beta, self.num_topics, self.vocab_size)
        c.tw_counts[:] = self.tw_counts
        c.t_counts[:] = self.t_counts
        return c
        

cdef class FirstMomentPLFilter:
    cdef GlobalParams model

    def __cinit__(self, GlobalParams model):
        self.model = model

    cdef likelihood(self, list sample):
        cdef numpy.double_t local_d_count, ll, s, p
        cdef numpy.double_t[:] local_dt_counts, x
        cdef numpy.uint_t i, j, t, w, num_words

        num_words = 0
        ll = 0.0
        local_d_count = 0.0
        local_dt_counts = numpy.zeros((self.model.num_topics,), dtype=numpy.double)
        x = numpy.zeros((self.model.num_topics,), dtype=numpy.double)
        for i in xrange(len(sample)):
            local_d_count = 0.0
            local_dt_counts[:] = 0.0
            for j in xrange(len(sample[i])):
                w = sample[i][j]
                for t in xrange(self.model.num_topics):
                    x[t] = (self.model.tw_counts[t, w] + self.model.beta) / (self.model.t_counts[t] + self.model.vocab_size * self.model.beta) * (local_dt_counts[t] + self.model.alpha) / (local_d_count + self.model.num_topics * self.model.alpha)
                s = numpy.sum(x)
                ll += numpy.log(s)
                for t in xrange(self.model.num_topics):
                    p = x[t] / s
                    local_dt_counts[t] += p
                    local_d_count += p
                num_words += 1
            if i % 100 == 0:
                PyErr_CheckSignals()

        print('Log-likelihood: %f' % ll)
        print('Perplexity:     %f' % numpy.exp(-ll / num_words))


cdef class ParticleFilter:
    cdef GlobalParams canonical_model
    cdef list models
    cdef numpy.double_t[:] weights
    cdef numpy.uint_t num_particles
    cdef numpy.double_t[:] pmf

    def __cinit__(self, GlobalParams init_model, numpy.uint_t num_particles):
        cdef numpy.uint_t i
        self.canonical_model = init_model
        self.models = []
        for i in xrange(num_particles):
            self.models.append(init_model.copy())
        self.weights = numpy.ones((num_particles,), dtype=numpy.double) / num_particles
        self.num_particles = num_particles
        self.pmf = numpy.zeros((init_model.num_topics,), dtype=numpy.double)

    cdef step(self, list doc):
        cdef numpy.uint_t i, z, t
        cdef GlobalParams model
        cdef numpy.double_t local_d_count, total_weight, prior
        cdef numpy.double_t[:] local_dt_counts

        local_d_count = 0.0
        local_dt_counts = numpy.zeros((self.canonical_model.num_topics,), dtype=numpy.double)
        for j in xrange(len(doc)):
            w = doc[j]
            for i in xrange(self.num_particles):
                model = <GlobalParams> self.models[i]
                prior = 0.0
                for t in xrange(self.canonical_model.num_topics):
                    prior += self.conditional_posterior(model, local_d_count, local_dt_counts, w, t)
                self.weights[i] *= prior
            total_weight = numpy.sum(self.weights)
            for i in xrange(self.num_particles):
                self.weights[i] /= total_weight
            for i in xrange(self.num_particles):
                model = <GlobalParams> self.models[i]
                z = self.sample_topic(model, local_d_count, local_dt_counts, w)
                model.tw_counts[z, w] += 1
                model.t_counts[z] += 1
                local_dt_counts[z] += 1
                local_d_count += 1

    cdef numpy.double_t conditional_posterior(self, GlobalParams model, numpy.double_t local_d_count, numpy.double_t[:] local_dt_counts, numpy.uint_t w, numpy.uint_t t):
        return (model.tw_counts[t, w] + self.canonical_model.beta) / (model.t_counts[t] + self.canonical_model.vocab_size * self.canonical_model.beta) * (local_dt_counts[t] + self.canonical_model.alpha) / (local_d_count + self.canonical_model.num_topics * self.canonical_model.alpha)

    cdef numpy.uint_t sample_topic(self, GlobalParams model, numpy.double_t local_d_count, numpy.double_t[:] local_dt_counts, numpy.uint_t w):
        cdef numpy.double_t prior, r
        cdef numpy.uint_t t

        prior = 0.0
        for t in xrange(self.canonical_model.num_topics):
            self.pmf[t] = self.conditional_posterior(model, local_d_count, local_dt_counts, w, t)
            prior += self.pmf[t]

        r = random.random() * prior
        for t in xrange(self.canonical_model.num_topics-1):
            if r < self.pmf[t]:
                return t
            self.pmf[t+1] += self.pmf[t]

        return self.canonical_model.num_topics - 1

    cdef GlobalParams max_posterior_model(self):
        return self.models[numpy.argmax(self.weights)]


cdef class GibbsSampler:
    cdef GlobalParams model
    cdef readonly numpy.uint_t[:, ::1] dt_counts
    cdef readonly numpy.uint_t[:] d_counts
    cdef numpy.double_t[:] pmf

    def __cinit__(self, GlobalParams model):
        self.model = model
        self.pmf = numpy.zeros((model.num_topics,), dtype=numpy.double)

    cdef numpy.double_t conditional_posterior(self, numpy.uint_t i, numpy.uint_t w, numpy.uint_t t):
        return (self.model.tw_counts[t, w] + self.model.beta) / (self.model.t_counts[t] + self.model.vocab_size * self.model.beta) * (self.dt_counts[i, t] + self.model.alpha) / (self.d_counts[i] + self.model.num_topics * self.model.alpha)

    cdef numpy.uint_t sample_topic(self, numpy.uint_t i, numpy.uint_t w):
        cdef numpy.double_t prior, r
        cdef numpy.uint_t t

        prior = 0.0
        for t in xrange(self.model.num_topics):
            self.pmf[t] = self.conditional_posterior(i, w, t)
            prior += self.pmf[t]

        r = random.random() * prior
        for t in xrange(self.model.num_topics-1):
            if r < self.pmf[t]:
                return t
            self.pmf[t+1] += self.pmf[t]

        return self.model.num_topics - 1

    cdef learn(self, list sample, numpy.uint_t num_iters):
        self.run(sample, num_iters, 1)

    cdef infer(self, list sample, numpy.uint_t num_iters):
        self.run(sample, num_iters, 0)

    cdef run(self, list sample, numpy.uint_t num_iters, bint update_model):
        cdef list assignments
        cdef numpy.uint_t t, i, j, w, m, z, num_docs

        num_docs = len(sample)

        assignments = []
        self.dt_counts = numpy.zeros((num_docs, self.model.num_topics), dtype=numpy.uint)
        self.d_counts = numpy.zeros((num_docs,), dtype=numpy.uint)

        for i in xrange(num_docs):
            for j in xrange(len(sample[i])):
                w = sample[i][j]
                z = random.randint(0, self.model.num_topics - 1)
                assignments.append(z)
                if update_model:
                    self.model.tw_counts[z, w] += 1
                    self.model.t_counts[z] += 1
                self.dt_counts[i, z] += 1
                self.d_counts[i] += 1
            if i % 100 == 0:
                PyErr_CheckSignals()

        for t in xrange(num_iters):
            m = 0
            for i in xrange(num_docs):
                for j in xrange(len(sample[i])):
                    w = sample[i][j]
                    z = assignments[m]
                    if update_model:
                        self.model.tw_counts[z, w] -= 1
                        self.model.t_counts[z] -= 1
                    self.dt_counts[i, z] -= 1
                    self.d_counts[i] -= 1
                    z = self.sample_topic(i, w)
                    assignments[m] = z
                    if update_model:
                        self.model.tw_counts[z, w] += 1
                        self.model.t_counts[z] += 1
                    self.dt_counts[i, z] += 1
                    self.d_counts[i] += 1
                    m += 1
                if i % 100 == 0:
                    PyErr_CheckSignals()
            sys.stdout.write('.')
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()


def run_lda(data_dir, categories, num_topics):
    cdef GibbsSampler init_gibbs_sampler, gibbs_sampler
    cdef GlobalParams model
    cdef FirstMomentPLFilter plfilter
    cdef ParticleFilter pf
    cdef numpy.uint_t i, reservoir_size, init_num_docs, init_num_iters, test_num_iters
    cdef numpy.double_t alpha, beta

    reservoir_size = 1000
    test_num_iters = 5
    alpha = 0.1
    beta = 0.1
    init_num_docs = 100
    init_num_iters = 100
    num_particles = 100

    dataset = Dataset(data_dir, set(categories))
    reservoir = pylowl.ValuedReservoirSampler(reservoir_size)
    model = GlobalParams(alpha, beta, num_topics, len(dataset.vocab))
    plfilter = FirstMomentPLFilter(model)

    def preprocess(doc_triple):
        return doc_triple[:2] + ([dataset.vocab[w] for w in doc_triple[2]],)

    test_data = [preprocess(t) for t in dataset.test_iterator()]
    test_sample = [t[2] for t in test_data]
    test_labels = [t[1] for t in test_data]

    init_sample = []
    init_labels = []

    train_sample = []
    train_labels = []

    pf = None

    i = 0
    for doc_triple in dataset.train_iterator():
        d = preprocess(doc_triple)
        if i < init_num_docs:
            init_sample.append(d[2])
            init_labels.append(d[1])
        elif i >= init_num_docs:
            if i == init_num_docs:
                print('Initializing on first %d docs' % i)
                print('Gibbs sampling with %d iters' % init_num_iters)
                init_gibbs_sampler = GibbsSampler(model)
                init_gibbs_sampler.learn(init_sample, init_num_iters)

                print(model.to_string(dataset.vocab, 20))
                print('In-sample NMI: %f' % nmi(init_labels, list(categories), init_gibbs_sampler.dt_counts, num_topics))
                gibbs_sampler = GibbsSampler(model)
                gibbs_sampler.infer(test_sample, test_num_iters)
                print('Out-of-sample NMI: %f' % nmi(test_labels, list(categories), gibbs_sampler.dt_counts, num_topics))
                plfilter.likelihood(test_sample)

                print('Creating particle filter on initialized model')
                pf = ParticleFilter(model, num_particles)
                train_labels = init_labels
            pf.step(d[2])
            train_labels.append(d[1])
            if i % 100 == 0:
                print('Doc %d' % i)
                #print(pf.max_posterior_model().to_string(dataset.vocab, 20))
                gibbs_sampler = GibbsSampler(pf.max_posterior_model())
                gibbs_sampler.infer(test_sample, test_num_iters)
                print('Out-of-sample NMI: %f' % nmi(test_labels, list(categories), gibbs_sampler.dt_counts, num_topics))
                #plfilter.likelihood(test_sample)
        i += 1

    print('Processed %d docs' % i)
    #print(pf.max_posterior_model().to_string(dataset.vocab, 20))
    gibbs_sampler = GibbsSampler(pf.max_posterior_model())
    gibbs_sampler.infer(test_sample, test_num_iters)
    print('Out-of-sample NMI: %f' % nmi(test_labels, list(categories), gibbs_sampler.dt_counts, num_topics))


if __name__ == '__main__':
    globals()[sys.argv[1]](*sys.argv[2:])
