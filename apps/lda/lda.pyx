#!/usr/bin/env python


from random import random, randint
from data import Dataset
from pylowl cimport ReservoirSampler, lowl_key, size_t
import sys
import numpy
import numpy.random
cimport numpy
from cpython.exc cimport PyErr_CheckSignals


# TODO check nmi


cdef object DEFAULT_PARAMS
DEFAULT_PARAMS = dict(
    reservoir_size = 1000,
    test_num_iters = 5,
    alpha = 0.1,
    beta = 0.1,
    ess_threshold = 20.0,
    init_num_docs = 100,
    init_num_iters = 100,
    num_particles = 100,
    rejuv_sample_size = 30,
    rejuv_mcmc_steps = 1,
    num_topics = 3,
    resample_propagate = False
)


cdef void shuffle(numpy.uint_t[::1] x):
    cdef numpy.uint_t n, i, j, temp
    n = len(x)
    for i in xrange(n - 1, 0, -1):
        j = randint(0, i)
        temp = x[i]
        x[i] = x[j]
        x[j] = temp


cdef numpy.uint_t[::1] sample_without_replacement(numpy.uint_t[::1] x,
        numpy.uint_t n):
    cdef numpy.uint_t[::1] samp
    if n < len(x):
        shuffle(x)
        samp = x[:n]
    else:
        samp = x
    return samp


cdef numpy.double_t perplexity(numpy.double_t likelihood, list sample):
    cdef numpy.uint_t num_words, i
    num_words = 0
    for i in xrange(len(sample)):
        num_words += len(sample[i])
    return numpy.exp(-likelihood / num_words)


cdef numpy.double_t conditional_posterior(
        numpy.uint_t[:] tw_counts, numpy.uint_t[::1] t_counts,
        numpy.uint_t[::1] dt_counts, numpy.uint_t d_count,
        numpy.double_t alpha, numpy.double_t beta,
        numpy.uint_t vocab_size, numpy.uint_t num_topics,
        numpy.uint_t t):
    return ((tw_counts[t] + beta) / (t_counts[t] + vocab_size * beta)
        * (dt_counts[t] + alpha) / (d_count + num_topics * alpha))


cdef numpy.double_t double_conditional_posterior(
        numpy.uint_t[:] tw_counts, numpy.uint_t[::1] t_counts,
        numpy.double_t[::1] dt_counts, numpy.double_t d_count,
        numpy.double_t alpha, numpy.double_t beta,
        numpy.uint_t vocab_size, numpy.uint_t num_topics,
        numpy.uint_t t):
    return ((tw_counts[t] + beta) / (t_counts[t] + vocab_size * beta)
        * (dt_counts[t] + alpha) / (d_count + num_topics * alpha))


cdef numpy.uint_t sample_topic(
        numpy.uint_t[:, ::1] tw_counts, numpy.uint_t[::1] t_counts,
        numpy.uint_t[::1] dt_counts, numpy.uint_t d_count,
        numpy.double_t alpha, numpy.double_t beta,
        numpy.uint_t vocab_size, numpy.uint_t num_topics,
        numpy.uint_t w, numpy.double_t[::1] pmf):
    cdef numpy.double_t prior, r
    cdef numpy.uint_t t

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


cdef numpy.double_t entropy2(long[::1] inferred_topics,
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
        long[::1] inferred_topics, numpy.uint_t num_topics):
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
                _mi += (count / n) * (numpy.log(count * n)
                    - numpy.log(marginal_count1 * marginal_count2))

    return _mi


cdef numpy.double_t nmi(list labels, list label_types,
        long[::1] inferred_topics, numpy.uint_t num_topics):
    cdef numpy.double_t _nmi

    _nmi = 2.0 * (mi(labels, label_types, inferred_topics, num_topics) /
        (entropy1(labels, label_types) + entropy2(inferred_topics, num_topics)))

    return _nmi


cdef class ParticleLabelStore:
    cdef list labels
    cdef numpy.uint_t num_particles, num_topics

    def __cinit__(self, numpy.uint_t num_particles, numpy.uint_t num_topics):
        cdef numpy.uint_t p
        self.num_particles = num_particles
        self.num_topics = num_topics
        self.labels = []
        for p in xrange(num_particles):
            self.labels.append([])

    cdef void append(self, numpy.uint_t p, numpy.uint_t label):
        self.labels[p].append(label)

    cdef void set(self, numpy.uint_t p, numpy.uint_t doc_idx,
            numpy.uint_t label):
        self.labels[p][doc_idx] = label

    cdef long compute_label(self, numpy.uint_t[::1] dt_counts):
        cdef numpy.uint_t t, max_t, count, max_count
        max_count = 0
        max_t = 0
        for t in xrange(self.num_topics):
            count = dt_counts[t]
            if count > max_count:
                max_count = count
                max_t = t
        return max_t

    cdef void recompute(self, ParticleFilterReservoirData rejuv_data):
        cdef numpy.uint_t[::1] dt_counts
        cdef numpy.uint_t p, i, j, doc_idx
        cdef long label

        for p in xrange(self.num_particles):
            for j in xrange(rejuv_data.occupied):
                i = rejuv_data.reservoir_idx_map[j]
                doc_idx = rejuv_data.doc_ids[j]
                dt_counts = rejuv_data.dt_counts[i, p, :] # TODO using the right index here?
                label = self.compute_label(dt_counts)
                self.set(p, doc_idx, label)

    cdef long[::1] label_view(self, numpy.uint_t p):
        cdef list particle_labels
        cdef long[::1] view
        cdef numpy.uint_t i
        particle_labels = self.labels[p]
        view = numpy.zeros((len(particle_labels),), dtype=numpy.long)
        for i in xrange(len(particle_labels)):
            view[i] = particle_labels[i]
        return view


cdef class GlobalModel:
    cdef numpy.double_t alpha, beta
    cdef numpy.uint_t num_topics, vocab_size
    cdef numpy.uint_t[:, ::1] tw_counts
    cdef numpy.uint_t[::1] t_counts

    def __cinit__(self, numpy.uint_t[:, ::1] tw_counts,
            numpy.uint_t[::1] t_counts,
            numpy.double_t alpha, numpy.double_t beta,
            numpy.uint_t num_topics, numpy.uint_t vocab_size):
        self.tw_counts = tw_counts
        self.t_counts = t_counts
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.vocab_size = vocab_size

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

    cdef numpy.double_t conditional_posterior(self,
            numpy.uint_t[::1] dt_counts, numpy.uint_t d_count,
            numpy.uint_t w, numpy.uint_t t):
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
        

cdef class FirstMomentPLFilter:
    cdef GlobalModel model

    def __cinit__(self, GlobalModel model):
        self.model = model

    cdef numpy.double_t conditional_posterior(self,
            numpy.double_t[::1] dt_counts, numpy.double_t d_count,
            numpy.uint_t w, numpy.uint_t t):
        return double_conditional_posterior(
            self.model.tw_counts[:, w], self.model.t_counts,
            dt_counts, d_count,
            self.model.alpha, self.model.beta,
            self.model.vocab_size, self.model.num_topics,
            t)

    cdef numpy.double_t likelihood(self, list sample):
        cdef numpy.double_t local_d_count, ll, s, p
        cdef numpy.double_t[::1] local_dt_counts, x
        cdef numpy.uint_t i, j, t, w

        ll = 0.0
        local_d_count = 0.0
        local_dt_counts = numpy.zeros(
            (self.model.num_topics,), dtype=numpy.double)
        x = numpy.zeros((self.model.num_topics,), dtype=numpy.double)
        for i in xrange(len(sample)):
            local_d_count = 0.0
            local_dt_counts[:] = 0.0
            for j in xrange(len(sample[i])):
                w = sample[i][j]
                for t in xrange(self.model.num_topics):
                    x[t] = self.conditional_posterior(
                        local_dt_counts, local_d_count, w, t)
                s = numpy.sum(x)
                ll += numpy.log(s)
                for t in xrange(self.model.num_topics):
                    p = x[t] / s
                    local_dt_counts[t] += p
                    local_d_count += p
            if i % 100 == 0:
                PyErr_CheckSignals()

        return ll


cdef class ParticleFilterReservoirData:
    cdef numpy.uint_t[::1] doc_ids
    cdef numpy.uint_t[::1] w
    cdef numpy.uint_t[:, ::1] z
    cdef numpy.uint_t[::1] reservoir_idx_map
    cdef numpy.uint_t[::1] r_reservoir_idx_map
    cdef numpy.uint_t[:, ::1] d_counts
    cdef numpy.uint_t[:, :, ::1] dt_counts
    cdef numpy.uint_t capacity, num_particles, num_topics, occupied

    def __cinit__(self, numpy.uint_t capacity, numpy.uint_t num_particles,
            numpy.uint_t num_topics):
        self.reservoir_idx_map = numpy.zeros((capacity,), dtype=numpy.uint)
        self.r_reservoir_idx_map = numpy.zeros((capacity,), dtype=numpy.uint)
        self.doc_ids = numpy.zeros((capacity,), dtype=numpy.uint)
        self.w = numpy.zeros((capacity,), dtype=numpy.uint)
        self.z = numpy.zeros((capacity, num_particles), dtype=numpy.uint)
        self.d_counts = numpy.zeros((capacity, num_particles), dtype=numpy.uint)
        self.dt_counts = numpy.zeros(
            (capacity, num_particles, num_topics), dtype=numpy.uint)
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


cdef class ParticleFilter:
    cdef GibbsSampler rejuv_sampler
    cdef ReservoirSampler rs
    cdef ParticleFilterReservoirData rejuv_data
    cdef ParticleLabelStore label_store
    cdef numpy.uint_t[::1] local_d_counts
    cdef numpy.uint_t[:, ::1] local_dt_counts
    cdef numpy.uint_t[:, :, ::1] tw_counts
    cdef numpy.uint_t[:, ::1] t_counts
    cdef numpy.double_t[::1] weights, pmf, resample_cmf
    cdef numpy.uint_t num_topics, vocab_size, num_particles, token_idx
    cdef numpy.uint_t rejuv_sample_size, rejuv_mcmc_steps
    cdef numpy.double_t alpha, beta, ess_threshold
    cdef bint resample_propagate

    def __cinit__(self, GlobalModel init_model, numpy.uint_t num_particles,
            numpy.double_t ess_threshold, ReservoirSampler rs,
            ParticleFilterReservoirData rejuv_data,
            numpy.uint_t rejuv_sample_size, numpy.uint_t rejuv_mcmc_steps,
            numpy.uint_t next_token_idx, ParticleLabelStore label_store,
            bint resample_propagate):
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

        self.local_dt_counts = numpy.zeros(
            (num_particles, init_model.num_topics),
            dtype=numpy.uint)
        self.local_d_counts = numpy.zeros(
            (num_particles,),
            dtype=numpy.uint)
        self.tw_counts = numpy.zeros(
            (num_particles, init_model.num_topics, init_model.vocab_size),
            dtype=numpy.uint)
        self.t_counts = numpy.zeros(
            (num_particles, init_model.num_topics),
            dtype=numpy.uint)
        for i in xrange(num_particles):
            self.tw_counts[i, :, :] = init_model.tw_counts
            self.t_counts[i, :] = init_model.t_counts

        self.rejuv_data = rejuv_data

        self.weights = (numpy.ones((num_particles,), dtype=numpy.double)
            / num_particles)
        self.pmf = numpy.zeros((init_model.num_topics,), dtype=numpy.double)
        self.resample_cmf = numpy.zeros((num_particles,), dtype=numpy.double)

        self.label_store = label_store

        self.resample_propagate = resample_propagate

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

        self.local_d_counts = numpy.zeros(
            (self.num_particles,), dtype=numpy.uint)
        self.local_dt_counts = numpy.zeros(
            (self.num_particles, self.num_topics), dtype=numpy.uint)

        for i in xrange(self.num_particles):
            self.label_store.append(i, 0)

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

            if self.resample_propagate:
                _ess = self.ess()
                print('resampling: ess %f; doc_idx %d, %d; token_idx %d'
                    % (_ess, doc_idx, j, self.token_idx))
                self.resample()

            for i in xrange(self.num_particles):
                z = self.sample_topic(i, w)
                zz[i] = z
                self.tw_counts[i, z, w] += 1
                self.t_counts[i, z] += 1
                self.local_dt_counts[i, z] += 1
                self.local_d_counts[i] += 1

            if not self.resample_propagate:
                inserted = self.rs._insert(self.token_idx, &reservoir_idx,
                    &ejected, &ejected_token_idx)
                if inserted:
                    self.rejuv_data.insert(reservoir_idx, doc_idx, w, zz,
                        self.local_d_counts, self.local_dt_counts)
                    k = self.rejuv_data.lookup(reservoir_idx)
                    self.local_d_counts = self.rejuv_data.d_counts[k,:]
                    self.local_dt_counts = self.rejuv_data.dt_counts[k,:,:]

                _ess = self.ess()
                if _ess < self.ess_threshold:
                    print('resampling: ess %f; doc_idx %d, %d; token_idx %d'
                        % (_ess, doc_idx, j, self.token_idx))
                    self.resample()
                    self.rejuvenate()
                    self.label_store.recompute(self.rejuv_data)

            self.token_idx += 1
            PyErr_CheckSignals()

        for i in xrange(self.num_particles):
            self.label_store.set(i, doc_idx,
                self.label_store.compute_label(self.local_dt_counts[i, :]))

    cdef void rejuvenate(self):
        cdef GlobalModel model
        cdef numpy.uint_t[::1] sample_candidates, sample
        cdef numpy.uint_t p, t, i, j, w, z

        sample_candidates = numpy.arange(self.rs.occupied(), dtype=numpy.uint)
        sample = sample_without_replacement(sample_candidates,
            self.rejuv_sample_size)

        for p in xrange(self.num_particles):
            for t in xrange(self.rejuv_mcmc_steps):
                for j in xrange(len(sample)):
                    rd_idx = sample[j]
                    w = self.rejuv_data.w[rd_idx]
                    z = self.rejuv_data.z[rd_idx, p]
                    i = self.rejuv_data.reservoir_idx_map[rd_idx]
                    self.tw_counts[p, z, w] -= 1
                    self.t_counts[p, z] -= 1
                    self.rejuv_data.dt_counts[i, p, z] -= 1
                    self.rejuv_data.d_counts[i, p] -= 1
                    z = sample_topic(
                        self.tw_counts[p, :, :], self.t_counts[p, :],
                        self.rejuv_data.dt_counts[i, p, :],
                        self.rejuv_data.d_counts[i, p],
                        self.alpha, self.beta,
                        self.vocab_size, self.num_topics,
                        w, self.pmf)
                    self.rejuv_data.z[rd_idx, p] = z
                    self.tw_counts[p, z, w] += 1
                    self.t_counts[p, z] += 1
                    self.rejuv_data.dt_counts[i, p, z] += 1
                    self.rejuv_data.d_counts[i, p] += 1

    cdef numpy.double_t conditional_posterior(self,
            numpy.uint_t p, numpy.uint_t w, numpy.uint_t t):
        return conditional_posterior(
            self.tw_counts[p, :, w], self.t_counts[p, :],
            self.local_dt_counts[p, :], self.local_d_counts[p],
            self.alpha, self.beta,
            self.vocab_size, self.num_topics,
            t)

    cdef numpy.uint_t sample_topic(self, numpy.uint_t p, numpy.uint_t w):
        return sample_topic(
            self.tw_counts[p, :, :], self.t_counts[p, :],
            self.local_dt_counts[p, :], self.local_d_counts[p],
            self.alpha, self.beta,
            self.vocab_size, self.num_topics,
            w, self.pmf)

    cdef model_for_particle(self, numpy.uint_t p):
        cdef GlobalModel model
        model = GlobalModel(self.tw_counts[p, :, :], self.t_counts[p, :],
            self.alpha, self.beta, self.num_topics, self.vocab_size)
        return model

    cdef numpy.uint_t max_posterior_particle(self):
        cdef numpy.uint_t p
        p = numpy.argmax(self.weights)
        return p

    cdef GlobalModel max_posterior_model(self):
        cdef GlobalModel model
        cdef numpy.uint_t p
        p = self.max_posterior_particle()
        model = self.model_for_particle(p)
        return model


cdef class GibbsSampler:
    cdef GlobalModel model
    cdef readonly numpy.uint_t[:, ::1] dt_counts
    cdef readonly numpy.uint_t[::1] d_counts
    cdef readonly list assignments
    cdef numpy.double_t[::1] pmf

    def __cinit__(self, GlobalModel model):
        self.model = model
        self.pmf = numpy.zeros((model.num_topics,), dtype=numpy.double)

    cdef numpy.uint_t sample_topic(self, numpy.uint_t i, numpy.uint_t w):
        return sample_topic(
            self.model.tw_counts, self.model.t_counts,
            self.dt_counts[i, :], self.d_counts[i],
            self.model.alpha, self.model.beta,
            self.model.vocab_size, self.model.num_topics,
            w, self.pmf)

    cdef learn(self, list sample, numpy.uint_t num_iters):
        self.run(sample, num_iters, 1)

    cdef infer(self, list sample, numpy.uint_t num_iters):
        self.run(sample, num_iters, 0)

    cdef run(self, list sample, numpy.uint_t num_iters, bint update_model):
        cdef numpy.uint_t t, i, j, w, m, z, num_docs

        num_docs = len(sample)

        self.assignments = []
        self.dt_counts = numpy.zeros(
            (num_docs, self.model.num_topics), dtype=numpy.uint)
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


cdef void eval_pf(numpy.uint_t num_topics, ParticleFilter pf,
        list test_sample, list test_labels, list train_labels,
        numpy.uint_t test_num_iters, list categories):
    cdef FirstMomentPLFilter plfilter
    cdef GlobalModel model
    cdef GibbsSampler gibbs_sampler
    cdef numpy.double_t ll
    cdef long[::1] inferred_topics

    model = pf.max_posterior_model()

    inferred_topics = pf.label_store.label_view(pf.max_posterior_particle())
    print('in-sample nmi: %f'
        % nmi(train_labels, categories, inferred_topics, num_topics))

    gibbs_sampler = GibbsSampler(model)
    gibbs_sampler.infer(test_sample, test_num_iters)
    inferred_topics = numpy.argmax(gibbs_sampler.dt_counts, 1)
    print('out-of-sample nmi: %f'
        % nmi(test_labels, categories, inferred_topics, num_topics))

    plfilter = FirstMomentPLFilter(model)
    ll = plfilter.likelihood(test_sample)
    print('out-of-sample log-likelihood: %f' % ll)
    print('out-of-sample perplexity: %f' % perplexity(ll, test_sample))


def create_pf(GlobalModel model, list init_sample,
        numpy.uint_t[:, ::1] dt_counts, numpy.uint_t[::1] d_counts,
        list assignments,
        numpy.uint_t reservoir_size, numpy.uint_t num_particles,
        numpy.uint_t num_topics, numpy.double_t ess_threshold,
        numpy.uint_t rejuv_sample_size, numpy.uint_t rejuv_mcmc_steps,
        bint resample_propagate):
    cdef ParticleFilter pf
    cdef ReservoirSampler rs
    cdef ParticleFilterReservoirData rejuv_data
    cdef ParticleLabelStore label_store
    cdef bint ejected, inserted
    cdef lowl_key ejected_token_idx
    cdef size_t reservoir_idx
    cdef numpy.uint_t ret, token_idx
    cdef numpy.uint_t[::1] particle_d_counts, zz
    cdef numpy.uint_t[:, ::1] particle_dt_counts

    label_store = ParticleLabelStore(num_particles, num_topics)
    rejuv_data = ParticleFilterReservoirData(reservoir_size, num_particles,
        num_topics)
    rs = ReservoirSampler()
    ret = rs.init(reservoir_size)
    token_idx = 0

    for doc_idx in xrange(len(init_sample)):
        for j in xrange(len(init_sample[doc_idx])):
            w = init_sample[doc_idx][j]
            inserted = rs._insert(token_idx, &reservoir_idx,
                &ejected, &ejected_token_idx)
            if inserted:
                zz = numpy.zeros(
                    (num_particles,), dtype=numpy.uint)
                particle_d_counts = numpy.zeros(
                    (num_particles,), dtype=numpy.uint)
                particle_dt_counts = numpy.zeros(
                    (num_particles, num_topics), dtype=numpy.uint)
                for p in xrange(num_particles):
                    zz[p] = assignments[token_idx]
                    particle_d_counts[p] = d_counts[doc_idx]
                    particle_dt_counts[p,:] = dt_counts[doc_idx,:]
                rejuv_data.insert(reservoir_idx, doc_idx, w, zz,
                    particle_d_counts, particle_dt_counts)
            token_idx += 1

        for p in xrange(num_particles):
            label_store.append(p,
                label_store.compute_label(dt_counts[doc_idx, :]))

    pf = ParticleFilter(model, num_particles, ess_threshold, rs, rejuv_data,
        rejuv_sample_size, rejuv_mcmc_steps, token_idx, label_store,
        resample_propagate)
    return pf


def run_lda(data_dir, categories, **kwargs):
    cdef GibbsSampler init_gibbs_sampler
    cdef GlobalModel model
    cdef FirstMomentPLFilter plfilter
    cdef ParticleFilter pf
    cdef numpy.uint_t i, j, doc_idx, num_tokens, p
    cdef numpy.uint_t[::1] t_counts
    cdef numpy.uint_t[:, ::1] tw_counts

    # load default params and override with contents of kwargs (if any)
    params = DEFAULT_PARAMS.copy()
    for (k, v) in kwargs.items():
        if k in params:
            params[k] = type(params[k])(v)
    print('params:')
    for (k, v) in params.items():
        print('\t%s = %s' % (k, str(v)))

    print('data dir: %s' % data_dir)

    print('categories:')
    for category in categories:
        print('\t%s' % category)

    dataset = Dataset(data_dir, set(categories))
    tw_counts = numpy.zeros(
        (params['num_topics'], len(dataset.vocab)), dtype=numpy.uint)
    t_counts = numpy.zeros((params['num_topics'],), dtype=numpy.uint)
    model = GlobalModel(tw_counts, t_counts, params['alpha'], params['beta'],
        params['num_topics'], len(dataset.vocab))
    plfilter = FirstMomentPLFilter(model)

    print('vocab size: %d' % len(dataset.vocab))

    def preprocess(doc_triple):
        # replace words with unique ids
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

        if i < params['init_num_docs']:
            # accumulate data for init gibbs sampler
            init_sample.append(d[2])
            init_labels.append(d[1])
        else:
            if i == params['init_num_docs']:
                # run init gibbs sampler on accumulated data, then
                # create pf from results
                print('initializing on first %d docs (%d tokens)'
                    % (i, num_tokens))
                print('gibbs sampling with %d iters' % params['init_num_iters'])
                init_gibbs_sampler = GibbsSampler(model)
                init_gibbs_sampler.learn(init_sample, params['init_num_iters'])

                print('creating particle filter on initialized model')
                pf = create_pf(model, init_sample, init_gibbs_sampler.dt_counts,
                    init_gibbs_sampler.d_counts, init_gibbs_sampler.assignments,
                    params['reservoir_size'], params['num_particles'],
                    params['num_topics'], params['ess_threshold'],
                    params['rejuv_sample_size'], params['rejuv_mcmc_steps'],
                    params['resample_propagate'])

                train_labels = init_labels
                eval_pf(params['num_topics'], pf,
                    test_sample, test_labels, train_labels,
                    params['test_num_iters'], list(categories))

                print(pf.max_posterior_model().to_string(dataset.vocab, 20))

            # process current document through pf
            pf.step(i, d[2])
            train_labels.append(d[1])
            print('doc: %d' % i)
            print('num words: %d' % len(d[2]))
            if i % 50 == 0:
                eval_pf(params['num_topics'], pf,
                    test_sample, test_labels, train_labels,
                    params['test_num_iters'], list(categories))

        num_tokens += len(d[2])
        i += 1

    if i <= params['init_num_docs']:
        # init_num_docs was really big; do Gibbs sampling and initialize
        # pf just so we can evaluate the model learned by Gibbs
        print('initializing on first %d docs (%d tokens)'
            % (i, num_tokens))
        print('gibbs sampling with %d iters' % params['init_num_iters'])
        init_gibbs_sampler = GibbsSampler(model)
        init_gibbs_sampler.learn(init_sample, params['init_num_iters'])

        print('creating particle filter on initialized model')
        pf = create_pf(model, init_sample, init_gibbs_sampler.dt_counts,
            init_gibbs_sampler.d_counts, init_gibbs_sampler.assignments,
            params['reservoir_size'], params['num_particles'],
            params['num_topics'], params['ess_threshold'],
            params['rejuv_sample_size'], params['rejuv_mcmc_steps'],
            params['resample_propagate'])

        train_labels = init_labels

    # end of run, do one last eval and print topics
    eval_pf(params['num_topics'], pf, test_sample, test_labels,
        train_labels, params['test_num_iters'], list(categories))

    print('processed %d docs (%d tokens)' % (i, num_tokens))
    print(pf.max_posterior_model().to_string(dataset.vocab, 20))
