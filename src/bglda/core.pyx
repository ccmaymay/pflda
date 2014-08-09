#!/usr/bin/env python

from random import random, randint, seed
from data import Dataset
from pylowl.core cimport ReservoirSampler, lowl_key, size_t
import sys
import numpy
cimport numpy
from cpython.exc cimport PyErr_CheckSignals
from libc.math cimport log, exp
from numpy import uint as np_uint, double as np_double, long as np_long, zeros, ones
from numpy cimport uint_t as np_uint_t, double_t as np_double_t, long_t as np_long_t


cdef object DEFAULT_PARAMS
DEFAULT_PARAMS = dict(
    # whether to print very verbose debugging information
    debug = False,

    # number of words to print per topic, when printing topics
    print_num_words_per_topic = 50,

    # whether to shuffle the data beforehand; note that for TNG at
    # least, data is pre-shuffled, so the effect of setting this option
    # to true is simply that every time the program is run you will
    # receive a different data ordering (NB: this requires loading
    # all data into memory!!)
    shuffle_data = False,

    # number of gibbs iterations (one per token) used when computing
    # out-of-sample nmi
    test_num_iters = 5,

    # dirichlet hyperparameter on topic distributions
    alpha = 0.1,

    # dirichlet hyperparameter on word distributions
    beta = 0.1,

    # number of docs to sample
    num_docs = -1,

    # number of gibbs sweeps to perform
    num_iters = 100,

    # prng seed for initialization: if non-negative, prng will be seeded
    # with this value before initialization and re-seeded randomly
    # afterward; if negative, do not seed prng (python chooses a random
    # state for us at start-up)
    init_seed = -1,

    # number of topics
    num_topics = 3,

    # number of particles used in left-to-right likelihood estimation
    ltr_num_particles = 20,

    # controls whether we use the left-to-right algorithm to perform
    # likelihood estimation, in addition to the first moment PL
    # approximation; note that this may increase runtime significantly
    ltr_eval = False,
)


# compute argmax of array of (assumed non-negative) doubles
cdef long udouble_argmax(np_double_t[::1] xx):
    cdef np_uint_t i, max_i, n
    cdef np_double_t x, max_x
    max_x = 0.0
    max_i = 0
    n = len(xx)
    for i in xrange(n):
        x = xx[i]
        if x > max_x:
            max_x = x
            max_i = i
    return max_i


# compute argmax of array of uints
cdef long uint_argmax(np_uint_t[::1] xx):
    cdef np_uint_t i, max_i, x, max_x, n
    max_x = 0
    max_i = 0
    n = len(xx)
    for i in xrange(n):
        x = xx[i]
        if x > max_x:
            max_x = x
            max_i = i
    return max_i


# swaps elements at indices i and j in the given array
cdef void swap(np_uint_t[::1] x, np_uint_t i, np_uint_t j):
    cdef np_uint_t temp

    temp = x[i]
    x[i] = x[j]
    x[j] = temp


# shuffle the given array
cdef void shuffle(np_uint_t[::1] x):
    cdef np_uint_t n, i, j
    n = len(x)
    for i in xrange(n - 1, 0, -1):
        j = randint(0, i)
        swap(x, i, j)


# sort an array of uints, in descending order, partially; that is,
# reorder the given array so that the largest `req` elements are sorted,
# descending, at the beginning of the list, and return an array of size
# `req` whose elements are the original indices of those elements
cdef np_uint_t[::1] reverse_sort_uint(np_uint_t[::1] x, np_uint_t req):
    cdef np_uint_t[::1] indices
    cdef np_uint_t n, i, j

    n = len(x)
    indices = zeros((n,), dtype=np_uint)
    for i in xrange(n):
        indices[i] = i

    for i in xrange(req):
        for j in xrange(i+1, n):
            if x[j] > x[i]:
                swap(x, i, j)
                swap(indices, i, j)

    return indices[:req]


# return an array of size n whose elements are sampled uniformly without
# replacement from [0, m); if m <= n, just return [0, m).
# (integers)
cdef np_uint_t[::1] sample_without_replacement(np_uint_t m, np_uint_t n):
    cdef np_uint_t[::1] samp, sample_candidates
    cdef np_uint_t j

    sample_candidates = zeros(m, dtype=np_uint)
    for j in xrange(m):
        sample_candidates[j] = j

    if n < m:
        shuffle(sample_candidates)
        samp = sample_candidates[:n]
    else:
        samp = sample_candidates
    return samp


# compute per-word perplexity given log-likelihood and list of docs
# (list of lists of word ids)
cdef np_double_t perplexity(np_double_t likelihood, list sample):
    cdef np_uint_t num_words, i
    num_words = 0
    for i in xrange(len(sample)):
        num_words += len(sample[i])
    return exp(-likelihood / num_words)


# compute value of unnormalized conditional topic assignment posterior
# at topic t
cdef np_double_t conditional_posterior(
        np_uint_t[:] tw_counts, np_uint_t[::1] t_counts,
        np_uint_t[::1] dt_counts, np_uint_t d_count,
        np_double_t alpha, np_double_t beta,
        np_uint_t vocab_size, np_uint_t num_topics,
        np_uint_t t):
    return ((tw_counts[t] + beta) / (t_counts[t] + vocab_size * beta)
        * (dt_counts[t] + alpha) / (d_count + num_topics * alpha))


# compute value of unnormalized conditional topic assignment posterior
# at topic t, where sufficient statistics are double type (fractional)
cdef np_double_t double_conditional_posterior(
        np_uint_t[:] tw_counts, np_uint_t[::1] t_counts,
        np_double_t[::1] dt_counts, np_double_t d_count,
        np_double_t alpha, np_double_t beta,
        np_uint_t vocab_size, np_uint_t num_topics,
        np_uint_t t):
    return ((tw_counts[t] + beta) / (t_counts[t] + vocab_size * beta)
        * (dt_counts[t] + alpha) / (d_count + num_topics * alpha))


# sample topic for word id w from conditional posterior, using pmf
# to store intermediate probabilities
cdef np_uint_t sample_topic(
        np_uint_t[:, ::1] tw_counts, np_uint_t[::1] t_counts,
        np_uint_t[::1] dt_counts, np_uint_t d_count,
        np_double_t alpha, np_double_t beta,
        np_uint_t vocab_size, np_uint_t num_topics,
        np_uint_t w, np_double_t[::1] pmf):
    cdef np_double_t prior, r
    cdef np_uint_t t

    prior = 0.0
    for t in xrange(num_topics):
        pmf[t] = conditional_posterior(
            tw_counts[:, w], t_counts,
            dt_counts, d_count,
            alpha, beta,
            vocab_size, num_topics,
            t)
        prior += pmf[t]

    r = random() * prior
    for t in xrange(num_topics-1):
        if r < pmf[t]:
            return t
        pmf[t+1] += pmf[t]

    return num_topics - 1


# structure containing hyperparameters and collapsed statistics of
# LDA model
cdef class GlobalModel:
    cdef np_double_t alpha, beta
    cdef np_uint_t num_topics, vocab_size
    cdef np_uint_t[:, ::1] tw_counts
    cdef np_uint_t[::1] t_counts

    def __cinit__(self, np_uint_t[:, ::1] tw_counts,
            np_uint_t[::1] t_counts,
            np_double_t alpha, np_double_t beta,
            np_uint_t num_topics, np_uint_t vocab_size):
        self.tw_counts = tw_counts
        self.t_counts = t_counts
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.vocab_size = vocab_size

    def to_string(self, vocab, num_words_per_topic):
        s = ''
        for t in range(self.num_topics):
            s += 'topic %d:' % t
            pp = [(word, self.tw_counts[t, w]) for (word, w) in vocab.items()]
            pp.sort(key=lambda p: p[1], reverse=True)
            i = 0
            for (word, count) in pp:
                if count > 0:
                    s += ' %s (%d)' % (word, count)
                i += 1
                if i >= num_words_per_topic:
                    break
            s += '\n'
        return s

    def to_string_raw(self):
        s = 'topics:\n'
        for t in range(self.num_topics):
            s += ' '
            pp = [(w, self.tw_counts[t, w]) for w in range(self.vocab_size)]
            pp.sort(key=lambda p: p[1], reverse=True)
            for (w, count) in pp:
                if count > 0:
                    s += ' %s (%d)' % (w, count)
                else:
                    break
            s += '\n'
        return s

    cdef np_double_t conditional_posterior(self,
            np_uint_t[::1] dt_counts, np_uint_t d_count,
            np_uint_t w, np_uint_t t):
        return conditional_posterior(
            self.tw_counts[:, w], self.t_counts,
            dt_counts, d_count,
            self.alpha, self.beta,
            self.vocab_size, self.num_topics,
            t)

    cdef GlobalModel copy(self):
        cdef GlobalModel c
        c = GlobalModel(self.tw_counts.copy(), self.t_counts.copy(),
            self.alpha, self.beta, self.num_topics, self.vocab_size)
        return c
        

# held-out likelihood estimator implementing particle learning filter
# of Scott and Baldridge (2013)
cdef class FirstMomentPLFilter:
    cdef GlobalModel model

    def __cinit__(self, GlobalModel model):
        self.model = model

    cdef np_double_t conditional_posterior(self,
            np_double_t[::1] dt_counts, np_double_t d_count,
            np_uint_t w, np_uint_t t):
        return double_conditional_posterior(
            self.model.tw_counts[:, w], self.model.t_counts,
            dt_counts, d_count,
            self.model.alpha, self.model.beta,
            self.model.vocab_size, self.model.num_topics,
            t)

    cdef np_double_t likelihood(self, list sample):
        cdef np_double_t local_d_count, ll, s, p
        cdef np_double_t[::1] local_dt_counts, x
        cdef np_uint_t i, j, t, w

        ll = 0.0
        local_d_count = 0.0
        local_dt_counts = zeros(
            (self.model.num_topics,), dtype=np_double)
        x = zeros((self.model.num_topics,), dtype=np_double)
        for i in xrange(len(sample)):
            local_d_count = 0.0
            local_dt_counts[:] = 0.0
            for j in xrange(len(sample[i])):
                w = sample[i][j]
                s = 0.0
                for t in xrange(self.model.num_topics):
                    x[t] = self.conditional_posterior(
                        local_dt_counts, local_d_count, w, t)
                    s += x[t]
                ll += log(s)
                for t in xrange(self.model.num_topics):
                    p = x[t] / s
                    local_dt_counts[t] += p
                    local_d_count += p
            if i % 100 == 0:
                PyErr_CheckSignals()

        return ll


# held-out likelihood estimator implementing left-to-right algorithm
# of Wallach et al. (2009)
cdef class LeftToRightEvaluator:
    cdef GlobalModel model
    cdef np_uint_t num_particles
    cdef np_double_t[::1] pmf
    cdef np_uint_t[:, ::1] local_dt_counts
    cdef np_uint_t[::1] local_d_counts

    def __cinit__(self, GlobalModel model, np_uint_t num_particles):
        self.model = model
        self.num_particles = num_particles
        self.pmf = zeros((model.num_topics,), dtype=np_double)
        self.local_d_counts = zeros(
            (self.num_particles,), dtype=np_uint)
        self.local_dt_counts = zeros(
            (self.num_particles, self.model.num_topics,), dtype=np_uint)

    cdef np_uint_t sample_topic(self, np_uint_t r, np_uint_t w):
        return sample_topic(
            self.model.tw_counts, self.model.t_counts,
            self.local_dt_counts[r, :], self.local_d_counts[r],
            self.model.alpha, self.model.beta,
            self.model.vocab_size, self.model.num_topics,
            w, self.pmf)

    cdef np_double_t conditional_posterior(self, np_uint_t r, np_uint_t w,
            np_uint_t t):
        return conditional_posterior(
            self.model.tw_counts[:, w], self.model.t_counts,
            self.local_dt_counts[r, :], self.local_d_counts[r],
            self.model.alpha, self.model.beta,
            self.model.vocab_size, self.model.num_topics,
            t)

    cdef np_double_t likelihood(self, list sample):
        cdef np_double_t ll, p
        cdef np_uint_t[:, ::1] z
        cdef np_uint_t i, j, k, r, t, m, w, max_doc_size

        max_doc_size = 0
        for i in xrange(len(sample)):
            m = len(sample[i])
            if m > max_doc_size:
                max_doc_size = m

        z = zeros(
            (self.num_particles, max_doc_size,), dtype=np_uint)

        ll = 0.0

        for i in xrange(len(sample)):
            self.local_d_counts[:] = 0
            self.local_dt_counts[:] = 0

            for j in xrange(len(sample[i])):
                p = 0.0

                for r in xrange(self.num_particles):
                    for k in xrange(j):
                        self.local_dt_counts[r, z[r, k]] -= 1
                        self.local_d_counts[r] -= 1

                        w = sample[i][k]
                        z[r, k] = self.sample_topic(r, w)

                        self.local_dt_counts[r, z[r, k]] += 1
                        self.local_d_counts[r] += 1

                    w = sample[i][j]
                    for t in xrange(self.model.num_topics):
                        p += self.conditional_posterior(r, w, t)
                    z[r, j] = self.sample_topic(r, w)
                    self.local_dt_counts[r, z[r, j]] += 1
                    self.local_d_counts[r] += 1

                ll += log(p / self.num_particles)

            if i % 100 == 0:
                PyErr_CheckSignals()

        return ll


# simple collapsed gibbs sampler for LDA
cdef class GibbsSampler:
    cdef GlobalModel model
    cdef readonly np_uint_t[:, ::1] dt_counts
    cdef readonly np_uint_t[::1] d_counts
    cdef readonly list assignments
    cdef np_double_t[::1] pmf

    def __cinit__(self, GlobalModel model):
        self.model = model
        self.pmf = zeros((model.num_topics,), dtype=np_double)

    cdef np_uint_t sample_topic(self, np_uint_t i, np_uint_t w):
        return sample_topic(
            self.model.tw_counts, self.model.t_counts,
            self.dt_counts[i, :], self.d_counts[i],
            self.model.alpha, self.model.beta,
            self.model.vocab_size, self.model.num_topics,
            w, self.pmf)

    cdef learn(self, list sample, np_uint_t num_iters):
        self.run(sample, num_iters, 1)

    cdef infer(self, list sample, np_uint_t num_iters):
        self.run(sample, num_iters, 0)

    cdef run(self, list sample, np_uint_t num_iters, bint update_model):
        self.init(sample, update_model)
        self.iterate(sample, num_iters, update_model)

    cdef init(self, list sample, bint update_model):
        cdef np_uint_t i, j, w, z, num_docs

        num_docs = len(sample)

        self.assignments = []
        self.dt_counts = zeros(
            (num_docs, self.model.num_topics), dtype=np_uint)
        self.d_counts = zeros((num_docs,), dtype=np_uint)

        for i in xrange(num_docs):
            for j in xrange(len(sample[i])):
                w = sample[i][j]
                z = randint(0, self.model.num_topics - 1)
                self.assignments.append(z)
                if update_model:
                    self.model.tw_counts[z, w] += 1
                    self.model.t_counts[z] += 1
                self.dt_counts[i, z] += 1
                self.d_counts[i] += 1
            if i % 100 == 0:
                PyErr_CheckSignals()

    cdef iterate(self, list sample, np_uint_t num_iters, bint update_model):
        cdef np_uint_t t, i, j, w, m, z, num_docs

        num_docs = len(sample)

        for t in xrange(num_iters):
            m = 0
            for i in xrange(num_docs):
                for j in xrange(len(sample[i])):
                    w = sample[i][j]
                    z = self.assignments[m]
                    if update_model:
                        self.model.tw_counts[z, w] -= 1
                        self.model.t_counts[z] -= 1
                    self.dt_counts[i, z] -= 1
                    self.d_counts[i] -= 1
                    z = self.sample_topic(i, w)
                    self.assignments[m] = z
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


# evaluate LDA model (learned by Gibbs sampler) according to a
# number of metrics, in-sample and out-of-sample, printing results
# to standard output
cdef void eval_model(np_uint_t num_topics, GibbsSampler train_gibbs_sampler,
        list test_sample,
        np_uint_t test_num_iters,
        bint ltr_eval,
        np_uint_t ltr_num_particles):
    cdef FirstMomentPLFilter plfilter
    cdef LeftToRightEvaluator ltr_evaluator
    cdef GlobalModel model
    cdef GibbsSampler gibbs_sampler
    cdef np_double_t ll
    cdef np_long_t[::1] inferred_topics

    model = train_gibbs_sampler.model

    plfilter = FirstMomentPLFilter(model)
    ll = plfilter.likelihood(test_sample)
    print('out-of-sample log-likelihood: %f' % ll)
    print('out-of-sample perplexity: %f' % perplexity(ll, test_sample))

    if ltr_eval:
        ltr_evaluator = LeftToRightEvaluator(model, ltr_num_particles)
        ll = ltr_evaluator.likelihood(test_sample)
        print('out-of-sample ltr log-likelihood: %f' % ll)
        print('out-of-sample ltr perplexity: %f' % perplexity(ll, test_sample))


# driver: learn LDA by collapsed Gibbs sampling
def run(data_dir, **kwargs):
    cdef GibbsSampler gs
    cdef GlobalModel model
    cdef np_uint_t doc_idx, num_tokens
    cdef np_uint_t[::1] t_counts
    cdef np_uint_t[:, ::1] tw_counts

    # load default params and override with contents of kwargs (if any)
    params = DEFAULT_PARAMS.copy()
    for (k, v) in kwargs.items():
        if k in params:
            params[k] = type(params[k])(v)

    print('params:')
    for (k, v) in params.items():
        print('\t%s = %s' % (k, str(v)))

    for k in kwargs:
        if k not in params:
            print('warning: unknown parameter %s' % k)

    print('data dir: %s' % data_dir)

    dataset = Dataset(data_dir, params['shuffle_data'])

    print('vocab size: %d' % len(dataset.vocab))

    def preprocess(doc):
        # replace words with unique ids
        return [dataset.vocab[w] for w in doc]

    test_sample = [preprocess(doc) for doc in dataset.test_iterator()]
    train_sample = []

    doc_idx = 0
    num_tokens = 0

    for doc in dataset.train_iterator():
        d = preprocess(doc)
        train_sample.append(d)
        doc_idx += 1
        num_tokens += len(d)
        if params['num_docs'] >= 0 and doc_idx == params['num_docs']:
            break

    if params['init_seed'] >= 0:
        print('seed: %u' % params['init_seed'])
        seed(params['init_seed'])
    else:
        rand_seed = randint(0, 1e9)
        print('seed: %u' % rand_seed)
        seed(rand_seed)

    tw_counts = zeros((params['num_topics'], len(dataset.vocab)), dtype=np_uint)
    t_counts = zeros((params['num_topics'],), dtype=np_uint)
    model = GlobalModel(tw_counts, t_counts, params['alpha'],
        params['beta'], params['num_topics'], len(dataset.vocab))

    print('gibbs sampling with %d iters' % params['num_iters'])
    gs = GibbsSampler(model)
    gs.init(train_sample, 1)

    print('iter: 0')
    eval_model(params['num_topics'], gs, test_sample,
        params['test_num_iters'],
        params['ltr_eval'],
        params['ltr_num_particles'])
    if params['debug']:
        print(model.to_string_raw())
    else:
        print(model.to_string(dataset.vocab, params['print_num_words_per_topic']))

    gs.iterate(train_sample, params['num_iters'], 1)

    eval_model(params['num_topics'], gs, test_sample,
        params['test_num_iters'],
        params['ltr_eval'],
        params['ltr_num_particles'])
    if params['debug']:
        print(model.to_string_raw())
    else:
        print(model.to_string(dataset.vocab,
            params['print_num_words_per_topic']))

    print('trained on %d docs (%d tokens)' % (doc_idx, num_tokens))
    if params['debug']:
        print(model.to_string_raw())
    else:
        print(model.to_string(dataset.vocab, params['print_num_words_per_topic']))
