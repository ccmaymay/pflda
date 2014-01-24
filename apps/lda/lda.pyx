#!/usr/bin/env python


from random import random, randint, sample as sample_random
from data import Dataset
from pylowl cimport ReservoirSampler, lowl_key, size_t
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
    cdef numpy.uint_t[::1] t_counts

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
        c.tw_counts[:, :] = self.tw_counts
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

        print('log-likelihood: %f' % ll)
        print('perplexity:     %f' % numpy.exp(-ll / num_words))


cdef class ParticleFilterReservoirData:
    cdef numpy.uint_t[::1] doc_ids
    cdef numpy.uint_t[::1] w
    cdef numpy.uint_t[:, ::1] z
    cdef numpy.uint_t[::1] reservoir_idx_map
    cdef numpy.uint_t[::1] r_reservoir_idx_map
    cdef numpy.uint_t[:, ::1] d_counts
    cdef numpy.uint_t[:, :, ::1] dt_counts
    cdef numpy.uint_t capacity, num_particles, num_topics, occupied

    def __cinit__(self, numpy.uint_t capacity, numpy.uint_t num_particles, numpy.uint_t num_topics):
        self.reservoir_idx_map = numpy.zeros((capacity,), dtype=numpy.uint)
        self.r_reservoir_idx_map = numpy.zeros((capacity,), dtype=numpy.uint)
        self.doc_ids = numpy.zeros((capacity,), dtype=numpy.uint)
        self.w = numpy.zeros((capacity,), dtype=numpy.uint)
        self.z = numpy.zeros((capacity, num_particles), dtype=numpy.uint)
        self.d_counts = numpy.zeros((capacity, num_particles), dtype=numpy.uint)
        self.dt_counts = numpy.zeros((capacity, num_particles, num_topics), dtype=numpy.uint)
        self.capacity = capacity
        self.num_particles = num_particles
        self.num_topics = num_topics
        self.occupied = 0

    cdef numpy.uint_t lookup(self, numpy.uint_t reservoir_idx):
        return self.reservoir_idx_map[reservoir_idx]

    cdef void insert(self, numpy.uint_t reservoir_idx,
            numpy.uint_t doc_idx, numpy.uint_t w, numpy.uint_t[::1] z,
            numpy.uint_t[::1] d_counts, numpy.uint_t[:, ::1] dt_counts):
        cdef numpy.uint_t i, j
        cdef bint doc_already_inserted

        self.w[reservoir_idx] = w
        self.z[reservoir_idx,:] = z

        if reservoir_idx < self.occupied:
            self.r_reservoir_idx_map[self.reservoir_idx_map[reservoir_idx]] -= 1

        doc_already_inserted = 0
        for i in xrange(self.occupied):
            if self.doc_ids[i] == doc_idx:
                j = self.reservoir_idx_map[i]
                self.reservoir_idx_map[reservoir_idx] = j
                self.r_reservoir_idx_map[j] += 1
                doc_already_inserted = 1
                break

        if not doc_already_inserted:
            for i in xrange(self.occupied):
                if self.r_reservoir_idx_map[i] == 0:
                    self.reservoir_idx_map[reservoir_idx] = i
                    self.r_reservoir_idx_map[i] += 1
                    self.d_counts[i, :] = d_counts
                    self.dt_counts[i, :, :] = dt_counts
                    break

        if reservoir_idx >= self.occupied:
            self.occupied += 1


# TODO: store list of all doc labels for in-sample eval
cdef class ParticleFilter:
    cdef GibbsSampler rejuv_sampler
    cdef ReservoirSampler rs
    cdef ParticleFilterReservoirData rejuv_data
    cdef numpy.uint_t[::1] local_d_counts
    cdef numpy.uint_t[:, ::1] local_dt_counts
    cdef numpy.uint_t[:, :, ::1] tw_counts
    cdef numpy.uint_t[:, ::1] t_counts
    cdef numpy.double_t[::1] weights, pmf, resample_cmf
    cdef numpy.uint_t num_topics, vocab_size, num_particles, token_idx
    cdef numpy.uint_t rejuv_sample_size, rejuv_mcmc_steps
    cdef numpy.double_t alpha, beta, ess_threshold

    def __cinit__(self, GlobalParams init_model, numpy.uint_t num_particles,
            numpy.double_t ess_threshold, ReservoirSampler rs,
            ParticleFilterReservoirData rejuv_data,
            numpy.uint_t rejuv_sample_size, numpy.uint_t rejuv_mcmc_steps,
            numpy.uint_t next_token_idx):
        cdef numpy.uint_t i

        self.alpha = init_model.alpha
        self.beta = init_model.beta
        self.num_topics = init_model.num_topics
        self.vocab_size = init_model.vocab_size
        self.num_particles = num_particles
        self.ess_threshold = ess_threshold
        self.rs = rs
        self.rejuv_sample_size = rejuv_sample_size
        self.rejuv_mcmc_steps = rejuv_mcmc_steps
        self.token_idx = next_token_idx

        self.local_dt_counts = numpy.zeros((num_particles, init_model.num_topics), dtype=numpy.uint)
        self.local_d_counts = numpy.zeros((num_particles,), dtype=numpy.uint)
        self.tw_counts = numpy.zeros((num_particles, init_model.num_topics, init_model.vocab_size), dtype=numpy.uint)
        self.t_counts = numpy.zeros((num_particles, init_model.num_topics), dtype=numpy.uint)
        for i in xrange(num_particles):
            self.tw_counts[i, :, :] = init_model.tw_counts
            self.t_counts[i, :] = init_model.t_counts

        self.rejuv_data = rejuv_data

        self.weights = numpy.ones((num_particles,), dtype=numpy.double) / num_particles
        self.pmf = numpy.zeros((init_model.num_topics,), dtype=numpy.double)
        self.resample_cmf = numpy.zeros((num_particles,), dtype=numpy.double)

    cdef numpy.double_t ess(self):
        cdef numpy.double_t total
        cdef numpy.uint_t i
        total = 0.0
        for i in xrange(self.num_particles):
            total += self.weights[i] * self.weights[i]
        return 1.0 / total

    cdef void resample(self):
        cdef numpy.uint_t i, j
        cdef numpy.uint_t[::1] ids_to_resample, filled_slots

        self.resample_cmf[0] = self.weights[0]
        for i in xrange(self.num_particles - 1):
            self.resample_cmf[i+1] = self.resample_cmf[i] + self.weights[i+1]

        ids_to_resample = numpy.zeros((self.num_particles,), dtype=numpy.uint)
        filled_slots = numpy.zeros((self.num_particles,), dtype=numpy.uint)
        for i in xrange(self.num_particles):
            j = self.sample_particle_num()
            if filled_slots[j] == 0:
                filled_slots[j] = 1
            else:
                ids_to_resample[j] += 1

        for i in xrange(self.num_particles):
            while ids_to_resample[i] > 0:
                j = 0
                while filled_slots[j] == 1:
                    j += 1
                self.tw_counts[j, :, :] = self.tw_counts[i, :, :]
                self.t_counts[j, :] = self.t_counts[i, :]
                self.local_dt_counts[j, :] = self.local_dt_counts[i, :]
                self.local_d_counts[j] = self.local_d_counts[i]
                filled_slots[j] = 1
                ids_to_resample[i] -= 1

        self.weights[:] = 1.0 / self.num_particles

    cdef numpy.uint_t sample_particle_num(self):
        cdef numpy.uint_t i
        cdef numpy.double_t r
        r = random()
        for i in xrange(self.num_particles):
            if r < self.resample_cmf[i]:
                return i
        return self.num_particles - 1

    cdef void step(self, numpy.uint_t doc_idx, list doc):
        cdef lowl_key ejected_token_idx
        cdef size_t reservoir_idx
        cdef numpy.uint_t k, z, i, t
        cdef numpy.uint_t[::1] zz
        cdef numpy.double_t total_weight, prior, _ess
        cdef bint inserted, ejected

        zz = numpy.zeros((self.num_particles,), dtype=numpy.uint)

        self.local_d_counts = numpy.zeros((self.num_particles,), dtype=numpy.uint)
        self.local_dt_counts = numpy.zeros((self.num_particles, self.num_topics), dtype=numpy.uint)

        for j in xrange(len(doc)):
            w = doc[j]

            for i in xrange(self.num_particles):
                prior = 0.0
                for t in xrange(self.num_topics):
                    prior += self.conditional_posterior(i, w, t)
                self.weights[i] *= prior
            total_weight = numpy.sum(self.weights)
            for i in xrange(self.num_particles):
                self.weights[i] /= total_weight
            for i in xrange(self.num_particles):
                z = self.sample_topic(i, w)
                zz[i] = z
                self.tw_counts[i, z, w] += 1
                self.t_counts[i, z] += 1
                self.local_dt_counts[i, z] += 1
                self.local_d_counts[i] += 1

            inserted = self.rs._insert(self.token_idx, &reservoir_idx, &ejected,
                &ejected_token_idx)
            if inserted:
                self.rejuv_data.insert(reservoir_idx, doc_idx, w, zz,
                    self.local_d_counts, self.local_dt_counts)
                k = self.rejuv_data.lookup(reservoir_idx)
                self.local_d_counts = self.rejuv_data.d_counts[k,:]
                self.local_dt_counts = self.rejuv_data.dt_counts[k,:,:]

            _ess = self.ess()
            if _ess < self.ess_threshold:
                print('ess is %f (doc_idx %d, %d; token_idx %d), resampling'
                    % (_ess, doc_idx, j, self.token_idx))
                self.resample()
                self.rejuvenate()

            self.token_idx += 1
            PyErr_CheckSignals()

    cdef void rejuvenate(self):
        cdef GlobalParams model
        cdef list sample
        cdef numpy.uint_t i

        sample = sample_random(self.rs.sample(), self.rejuv_sample_size)

        for i in xrange(self.num_particles):
            model = self.model_for_particle(i)
            self.rejuv_sampler = GibbsSampler(model)
#            self.rejuv_sampler.learn_pf_rejuv(sample, i, self.rejuv_mcmc_steps)

    cdef numpy.double_t conditional_posterior(self, numpy.uint_t p, numpy.uint_t w, numpy.uint_t t):
        return (self.tw_counts[p, t, w] + self.beta) / (self.t_counts[p, t] + self.vocab_size * self.beta) * (self.local_dt_counts[p, t] + self.alpha) / (self.local_d_counts[p] + self.num_topics * self.alpha)

    cdef numpy.uint_t sample_topic(self, numpy.uint_t p, numpy.uint_t w):
        cdef numpy.double_t prior, r
        cdef numpy.uint_t t

        prior = 0.0
        for t in xrange(self.num_topics):
            self.pmf[t] = self.conditional_posterior(p, w, t)
            prior += self.pmf[t]

        r = random() * prior
        for t in xrange(self.num_topics-1):
            if r < self.pmf[t]:
                return t
            self.pmf[t+1] += self.pmf[t]

        return self.num_topics - 1

    cdef model_for_particle(self, numpy.uint_t p):
        cdef GlobalParams model
        model = GlobalParams(self.alpha, self.beta, self.num_topics, self.vocab_size)
        model.tw_counts[:, :] = self.tw_counts[p, :, :]
        model.t_counts[:] = self.t_counts[p, :]
        return model

    cdef GlobalParams max_posterior_model(self):
        cdef GlobalParams model
        cdef numpy.uint_t p
        p = numpy.argmax(self.weights)
        model = self.model_for_particle(p)
        return model


cdef class GibbsSampler:
    cdef GlobalParams model
    cdef readonly numpy.uint_t[:, ::1] dt_counts
    cdef readonly numpy.uint_t[::1] d_counts
    cdef readonly list assignments
    cdef numpy.double_t[::1] pmf

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

        r = random() * prior
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
        cdef numpy.uint_t t, i, j, w, m, z, num_docs

        num_docs = len(sample)

        self.assignments = []
        self.dt_counts = numpy.zeros((num_docs, self.model.num_topics), dtype=numpy.uint)
        self.d_counts = numpy.zeros((num_docs,), dtype=numpy.uint)

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

#    cdef learn_pf_rejuv(self, list sample, numpy.uint_t p, numpy.uint_t num_iters):
#        cdef numpy.uint_t t, i, j, jj, w, z, num_docs
#        cdef object doc_idx_map
#
#        doc_idx_map = dict()
#        for j in xrange(len(sample)):
#            doc_idx = sample[j][0]
#            if doc_idx not in doc_idx_map:
#                doc_idx_map[doc_idx] = len(doc_idx_map)
#
#        num_docs = len(doc_idx_map)
#        self.dt_counts = numpy.zeros((num_docs, self.model.num_topics), dtype=numpy.uint)
#        self.d_counts = numpy.zeros((num_docs,), dtype=numpy.uint)
#
#        for j in xrange(len(sample)):
#            doc_idx = sample[j][0]
#            particle_reservoir_data = sample[j][2][p]
#            i = doc_idx_map[doc_idx]
#            # should be okay wthat we overwrite these... should all be
#            # the same... ugh
#            self.d_counts[i] = particle_reservoir_data[1]
#            for jj in xrange(self.model.num_topics):
#                self.dt_counts[i, jj] = particle_reservoir_data[2][jj]
#
#        for t in xrange(num_iters):
#            for j in xrange(len(sample)):
#                doc_idx = sample[j][0]
#                w = sample[j][1]
#                particle_reservoir_data = sample[j][2][p]
#                z = particle_reservoir_data[0]
#                i = doc_idx_map[doc_idx]
#                self.model.tw_counts[z, w] -= 1
#                self.model.t_counts[z] -= 1
#                self.dt_counts[i, z] -= 1
#                self.d_counts[i] -= 1
#                z = self.sample_topic(i, w)
#                particle_reservoir_data[0] = z
#                self.model.tw_counts[z, w] += 1
#                self.model.t_counts[z] += 1
#                self.dt_counts[i, z] += 1
#                self.d_counts[i] += 1
#                if j % 100 == 0:
#                    PyErr_CheckSignals()
#
#        # TODO these doc vects are not synchronized with those
#        # used by pf step...
#        for j in xrange(len(sample)):
#            doc_idx = sample[j][0]
#            particle_reservoir_data = sample[j][2][p]
#            i = doc_idx_map[doc_idx]
#            particle_reservoir_data[1] = self.d_counts[i]
#            for jj in xrange(self.model.num_topics):
#                particle_reservoir_data[2][jj] = self.dt_counts[i, jj]


def run_lda(data_dir, categories, num_topics):
    cdef GibbsSampler init_gibbs_sampler, gibbs_sampler
    cdef GlobalParams model
    cdef FirstMomentPLFilter plfilter
    cdef ParticleFilter pf
    cdef ReservoirSampler rs
    cdef ParticleFilterReservoirData rejuv_data
    cdef numpy.uint_t i, j, doc_idx, num_tokens, reservoir_size, init_num_docs
    cdef numpy.uint_t init_num_iters, test_num_iters, rejuv_sample_size, p
    cdef numpy.uint_t rejuv_mcmc_steps, ret, token_idx
    cdef numpy.uint_t[::1] particle_d_counts, dt_counts, zz
    cdef numpy.uint_t[:, ::1] particle_dt_counts
    cdef numpy.double_t alpha, beta, ess_threshold
    cdef bint ejected, inserted
    cdef lowl_key ejected_token_idx
    cdef size_t reservoir_idx

    reservoir_size = 1000
    test_num_iters = 5
    alpha = 0.1
    beta = 0.1
    ess_threshold = 20.0
    init_num_docs = 100
    init_num_iters = 100
    num_particles = 100
    rejuv_sample_size = 10
    rejuv_mcmc_steps = 30

    dataset = Dataset(data_dir, set(categories))
    model = GlobalParams(alpha, beta, num_topics, len(dataset.vocab))
    plfilter = FirstMomentPLFilter(model)

    print('vocab size: %d' % len(dataset.vocab))

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
    num_tokens = 0
    for doc_triple in dataset.train_iterator():
        d = preprocess(doc_triple)
        if i < init_num_docs:
            init_sample.append(d[2])
            init_labels.append(d[1])
        elif i >= init_num_docs:
            if i == init_num_docs:
                print('initializing on first %d docs (%d tokens)' % (i, num_tokens))
                print('gibbs sampling with %d iters' % init_num_iters)
                init_gibbs_sampler = GibbsSampler(model)
                init_gibbs_sampler.learn(init_sample, init_num_iters)

                print(model.to_string(dataset.vocab, 20))
                print('in-sample nmi: %f' % nmi(init_labels, list(categories), init_gibbs_sampler.dt_counts, num_topics))
                gibbs_sampler = GibbsSampler(model)
                gibbs_sampler.infer(test_sample, test_num_iters)
                print('out-of-sample nmi: %f' % nmi(test_labels, list(categories), gibbs_sampler.dt_counts, num_topics))
                #plfilter.likelihood(test_sample)

                print('creating particle filter on initialized model')
                rejuv_data = ParticleFilterReservoirData(reservoir_size,
                    num_particles, num_topics)
                rs = ReservoirSampler(reservoir_size)
                ret = rs.init(reservoir_size)
                token_idx = 0
                for doc_idx in xrange(len(init_sample)):
                    for j in xrange(len(init_sample[doc_idx])):
                        w = init_sample[doc_idx][j]
                        inserted = rs._insert(token_idx, &reservoir_idx,
                            &ejected, &ejected_token_idx)
                        if inserted:
                            zz = numpy.zeros((num_particles,), dtype=numpy.uint)
                            particle_d_counts = numpy.zeros((num_particles,), dtype=numpy.uint)
                            particle_dt_counts = numpy.zeros((num_particles, num_topics), dtype=numpy.uint)
                            for p in xrange(num_particles):
                                zz[p] = init_gibbs_sampler.assignments[token_idx]
                                particle_d_counts[p] = init_gibbs_sampler.d_counts[doc_idx]
                                particle_dt_counts[p,:] = init_gibbs_sampler.dt_counts[doc_idx,:]
                            rejuv_data.insert(reservoir_idx, doc_idx, w, zz,
                                particle_d_counts, particle_dt_counts)
                        token_idx += 1
                pf = ParticleFilter(model, num_particles, ess_threshold,
                    rs, rejuv_data, rejuv_sample_size, rejuv_mcmc_steps,
                    token_idx)
                train_labels = init_labels
            pf.step(i, d[2])
            train_labels.append(d[1])
            if i % 100 == 0:
                print('doc %d' % i)
                #print(pf.max_posterior_model().to_string(dataset.vocab, 20))
                gibbs_sampler = GibbsSampler(pf.max_posterior_model())
                gibbs_sampler.infer(test_sample, test_num_iters)
                print('out-of-sample nmi: %f' % nmi(test_labels, list(categories), gibbs_sampler.dt_counts, num_topics))
                #plfilter.likelihood(test_sample)
        num_tokens += len(d[2])
        i += 1

    print('processed %d docs (%d tokens)' % (i, num_tokens))
    print(pf.max_posterior_model().to_string(dataset.vocab, 20))
    gibbs_sampler = GibbsSampler(pf.max_posterior_model())
    gibbs_sampler.infer(test_sample, test_num_iters)
    print('out-of-sample nmi: %f' % nmi(test_labels, list(categories), gibbs_sampler.dt_counts, num_topics))


if __name__ == '__main__':
    globals()[sys.argv[1]](*sys.argv[2:])
