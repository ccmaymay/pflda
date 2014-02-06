#!/usr/bin/env python

from random import random, randint, seed
from data import Dataset
from pylowl cimport ReservoirSampler, lowl_key, size_t
import sys
import numpy
cimport numpy
from cpython.exc cimport PyErr_CheckSignals
from libc.math cimport log, exp
from numpy import uint as np_uint, double as np_double, long as np_long, zeros, ones
from numpy cimport uint_t as np_uint_t, double_t as np_double_t, long_t as np_long_t


cdef object DEFAULT_PARAMS
DEFAULT_PARAMS = dict(
    reservoir_size = 1000,
    test_num_iters = 5,
    alpha = 0.1,
    beta = 0.1,
    ess_threshold = 20.0,
    init_num_docs = 100,
    init_num_iters = 100,
    init_tune_seed = -1,
    init_tune_train_frac = 0.8,
    init_tune_num_cv_folds = 1,
    init_tune_num_runs = 1,
    init_tune_eval_nmi = False,
    num_particles = 100,
    rejuv_sample_size = 30,
    rejuv_mcmc_steps = 1,
    num_topics = 3,
    resample_propagate = False
)


cdef long double_argmax(np_double_t[::1] xx, np_uint_t n):
    cdef np_uint_t i, max_i
    cdef np_double_t x, max_x
    max_x = 0.0
    max_i = 0
    for i in xrange(n):
        x = xx[i]
        if x > max_x:
            max_x = x
            max_i = i
    return max_i


cdef long uint_argmax(np_uint_t[::1] xx, np_uint_t n):
    cdef np_uint_t i, max_i, x, max_x
    max_x = 0
    max_i = 0
    for i in xrange(n):
        x = xx[i]
        if x > max_x:
            max_x = x
            max_i = i
    return max_i


cdef void shuffle(np_uint_t[::1] x):
    cdef np_uint_t n, i, j, temp
    n = len(x)
    for i in xrange(n - 1, 0, -1):
        j = randint(0, i)
        temp = x[i]
        x[i] = x[j]
        x[j] = temp


cdef np_uint_t[::1] sample_without_replacement(np_uint_t[::1] x,
        np_uint_t n):
    cdef np_uint_t[::1] samp
    if n < len(x):
        shuffle(x)
        samp = x[:n]
    else:
        samp = x
    return samp


cdef np_double_t perplexity(np_double_t likelihood, list sample):
    cdef np_uint_t num_words, i
    num_words = 0
    for i in xrange(len(sample)):
        num_words += len(sample[i])
    return exp(-likelihood / num_words)


cdef np_double_t conditional_posterior(
        np_uint_t[:] tw_counts, np_uint_t[::1] t_counts,
        np_uint_t[::1] dt_counts, np_uint_t d_count,
        np_double_t alpha, np_double_t beta,
        np_uint_t vocab_size, np_uint_t num_topics,
        np_uint_t t):
    return ((tw_counts[t] + beta) / (t_counts[t] + vocab_size * beta)
        * (dt_counts[t] + alpha) / (d_count + num_topics * alpha))


cdef np_double_t double_conditional_posterior(
        np_uint_t[:] tw_counts, np_uint_t[::1] t_counts,
        np_double_t[::1] dt_counts, np_double_t d_count,
        np_double_t alpha, np_double_t beta,
        np_uint_t vocab_size, np_uint_t num_topics,
        np_uint_t t):
    return ((tw_counts[t] + beta) / (t_counts[t] + vocab_size * beta)
        * (dt_counts[t] + alpha) / (d_count + num_topics * alpha))


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


cdef np_double_t entropy1(list labels, list label_types):
    cdef np_uint_t i, j
    cdef np_double_t n, count, p, _entropy

    _entropy = 0.0
    n = float(len(labels))
    for i in xrange(len(label_types)):
        count = 0.0
        for j in xrange(len(labels)):
            if labels[j] == label_types[i]:
                count += 1.0
        p = count / n
        if p > 0.0:
            _entropy += -p * log(p)

    return _entropy


cdef np_double_t entropy2(np_long_t[::1] inferred_topics,
        np_uint_t num_topics):
    cdef np_uint_t i, t
    cdef np_double_t n, count, p, _entropy

    _entropy = 0.0
    n = float(inferred_topics.shape[0])
    for t in xrange(num_topics):
        count = 0.0
        for i in xrange(inferred_topics.shape[0]):
            if inferred_topics[i] == t:
                count += 1.0
        p = count / n
        if p > 0.0:
            _entropy += -p * log(p)

    return _entropy


cdef np_double_t mi(list labels, list label_types,
        np_long_t[::1] inferred_topics, np_uint_t num_topics):
    cdef np_uint_t i, t, j
    cdef np_double_t n, count, marginal_count1, marginal_count2, _mi

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
                _mi += (count / n) * (log(count * n)
                    - log(marginal_count1 * marginal_count2))

    return _mi


cdef np_double_t nmi(list labels, list label_types,
        np_long_t[::1] inferred_topics, np_uint_t num_topics):
    cdef np_double_t _nmi

    _nmi = 2.0 * (mi(labels, label_types, inferred_topics, num_topics) /
        (entropy1(labels, label_types) + entropy2(inferred_topics, num_topics)))

    return _nmi


cdef class ParticleLabelStore:
    cdef list labels
    cdef np_uint_t num_particles, num_topics

    def __cinit__(self, np_uint_t num_particles, np_uint_t num_topics):
        cdef np_uint_t p
        self.num_particles = num_particles
        self.num_topics = num_topics
        self.labels = []
        for p in xrange(num_particles):
            self.labels.append([])

    cdef void append(self, np_uint_t p, np_uint_t label):
        self.labels[p].append(label)

    cdef void set(self, np_uint_t p, np_uint_t doc_idx,
            np_uint_t label):
        self.labels[p][doc_idx] = label

    cdef long compute_label(self, np_uint_t[::1] dt_counts):
        return uint_argmax(dt_counts, self.num_topics)

    cdef void recompute(self, ParticleFilterReservoirData rejuv_data):
        cdef np_uint_t[::1] dt_counts
        cdef np_uint_t p, i, j, doc_idx
        cdef long label

        for p in xrange(self.num_particles):
            for j in xrange(rejuv_data.occupied):
                i = rejuv_data.reservoir_idx_map[j]
                doc_idx = rejuv_data.doc_ids[j]
                dt_counts = rejuv_data.dt_counts[i, p, :] # TODO using the right index here?
                label = self.compute_label(dt_counts)
                self.set(p, doc_idx, label)

    cdef np_long_t[::1] label_view(self, np_uint_t p):
        cdef list particle_labels
        cdef np_long_t[::1] view
        cdef np_uint_t i
        particle_labels = self.labels[p]
        view = zeros((len(particle_labels),), dtype=np_long)
        for i in xrange(len(particle_labels)):
            view[i] = particle_labels[i]
        return view


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


cdef class ParticleFilterReservoirData:
    cdef np_uint_t[::1] doc_ids
    cdef np_uint_t[::1] w
    cdef np_uint_t[:, ::1] z
    cdef np_uint_t[::1] reservoir_idx_map
    cdef np_uint_t[::1] r_reservoir_idx_map
    cdef np_uint_t[:, ::1] d_counts
    cdef np_uint_t[:, :, ::1] dt_counts
    cdef np_uint_t capacity, num_particles, num_topics, occupied

    def __cinit__(self, np_uint_t capacity, np_uint_t num_particles,
            np_uint_t num_topics):
        self.reservoir_idx_map = zeros((capacity,), dtype=np_uint)
        self.r_reservoir_idx_map = zeros((capacity,), dtype=np_uint)
        self.doc_ids = zeros((capacity,), dtype=np_uint)
        self.w = zeros((capacity,), dtype=np_uint)
        self.z = zeros((capacity, num_particles), dtype=np_uint)
        self.d_counts = zeros((capacity, num_particles), dtype=np_uint)
        self.dt_counts = zeros(
            (capacity, num_particles, num_topics), dtype=np_uint)
        self.capacity = capacity
        self.num_particles = num_particles
        self.num_topics = num_topics
        self.occupied = 0

    cdef np_uint_t lookup(self, np_uint_t reservoir_idx):
        return self.reservoir_idx_map[reservoir_idx]

    cdef void insert(self, np_uint_t reservoir_idx,
            np_uint_t doc_idx, np_uint_t w, np_uint_t[::1] z,
            np_uint_t[::1] d_counts, np_uint_t[:, ::1] dt_counts):
        cdef np_uint_t i, j
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
    cdef np_uint_t[::1] local_d_counts
    cdef np_uint_t[:, ::1] local_dt_counts
    cdef np_uint_t[:, :, ::1] tw_counts
    cdef np_uint_t[:, ::1] t_counts
    cdef np_double_t[::1] weights, pmf, resample_cmf
    cdef np_uint_t num_topics, vocab_size, num_particles, token_idx
    cdef np_uint_t rejuv_sample_size, rejuv_mcmc_steps
    cdef np_double_t alpha, beta, ess_threshold
    cdef bint resample_propagate

    def __cinit__(self, GlobalModel init_model, np_uint_t num_particles,
            np_double_t ess_threshold, ReservoirSampler rs,
            ParticleFilterReservoirData rejuv_data,
            np_uint_t rejuv_sample_size, np_uint_t rejuv_mcmc_steps,
            np_uint_t next_token_idx, ParticleLabelStore label_store,
            bint resample_propagate):
        cdef np_uint_t i

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

        self.local_dt_counts = zeros(
            (num_particles, init_model.num_topics),
            dtype=np_uint)
        self.local_d_counts = zeros(
            (num_particles,),
            dtype=np_uint)
        self.tw_counts = zeros(
            (num_particles, init_model.num_topics, init_model.vocab_size),
            dtype=np_uint)
        self.t_counts = zeros(
            (num_particles, init_model.num_topics),
            dtype=np_uint)
        for i in xrange(num_particles):
            self.tw_counts[i, :, :] = init_model.tw_counts
            self.t_counts[i, :] = init_model.t_counts

        self.rejuv_data = rejuv_data

        self.weights = (ones((num_particles,), dtype=np_double)
            / num_particles)
        self.pmf = zeros((init_model.num_topics,), dtype=np_double)
        self.resample_cmf = zeros((num_particles,), dtype=np_double)

        self.label_store = label_store

        self.resample_propagate = resample_propagate

    cdef np_double_t ess(self):
        cdef np_double_t total
        cdef np_uint_t i
        total = 0.0
        for i in xrange(self.num_particles):
            total += self.weights[i] * self.weights[i]
        return 1.0 / total

    cdef void resample(self):
        cdef np_uint_t i, j
        cdef np_uint_t[::1] ids_to_resample, filled_slots

        self.resample_cmf[0] = self.weights[0]
        for i in xrange(self.num_particles - 1):
            self.resample_cmf[i+1] = self.resample_cmf[i] + self.weights[i+1]

        ids_to_resample = zeros((self.num_particles,), dtype=np_uint)
        filled_slots = zeros((self.num_particles,), dtype=np_uint)
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

    cdef np_uint_t sample_particle_num(self):
        cdef np_uint_t i
        cdef np_double_t r
        r = random()
        for i in xrange(self.num_particles):
            if r < self.resample_cmf[i]:
                return i
        return self.num_particles - 1

    cdef void step(self, np_uint_t doc_idx, list doc):
        cdef lowl_key ejected_token_idx
        cdef size_t reservoir_idx
        cdef np_uint_t k, z, i, t
        cdef np_uint_t[::1] zz
        cdef np_double_t total_weight, prior, _ess
        cdef bint inserted, ejected

        zz = zeros((self.num_particles,), dtype=np_uint)

        self.local_d_counts = zeros(
            (self.num_particles,), dtype=np_uint)
        self.local_dt_counts = zeros(
            (self.num_particles, self.num_topics), dtype=np_uint)

        for i in xrange(self.num_particles):
            self.label_store.append(i, 0)

        for j in xrange(len(doc)):
            w = doc[j]

            total_weight = 0
            for i in xrange(self.num_particles):
                prior = 0.0
                for t in xrange(self.num_topics):
                    prior += self.conditional_posterior(i, w, t)
                self.weights[i] *= prior
                total_weight += self.weights[i]
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
        cdef np_uint_t[::1] sample_candidates, sample
        cdef np_uint_t p, t, i, j, w, z

        sample_candidates = zeros(self.rs.occupied(), dtype=np_uint)
        for j in xrange(self.rs.occupied()):
            sample_candidates[j] = j
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

    cdef np_double_t conditional_posterior(self,
            np_uint_t p, np_uint_t w, np_uint_t t):
        return conditional_posterior(
            self.tw_counts[p, :, w], self.t_counts[p, :],
            self.local_dt_counts[p, :], self.local_d_counts[p],
            self.alpha, self.beta,
            self.vocab_size, self.num_topics,
            t)

    cdef np_uint_t sample_topic(self, np_uint_t p, np_uint_t w):
        return sample_topic(
            self.tw_counts[p, :, :], self.t_counts[p, :],
            self.local_dt_counts[p, :], self.local_d_counts[p],
            self.alpha, self.beta,
            self.vocab_size, self.num_topics,
            w, self.pmf)

    cdef model_for_particle(self, np_uint_t p):
        cdef GlobalModel model
        model = GlobalModel(self.tw_counts[p, :, :], self.t_counts[p, :],
            self.alpha, self.beta, self.num_topics, self.vocab_size)
        return model

    cdef np_uint_t max_posterior_particle(self):
        cdef np_uint_t p
        p = double_argmax(self.weights, self.num_particles)
        return p

    cdef GlobalModel max_posterior_model(self):
        cdef GlobalModel model
        cdef np_uint_t p
        p = self.max_posterior_particle()
        model = self.model_for_particle(p)
        return model


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
        cdef np_uint_t t, i, j, w, m, z, num_docs

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


cdef np_long_t[::1] infer_topics(np_uint_t[:, ::1] dt_counts,
        np_uint_t num_docs, np_uint_t num_topics):
    cdef np_long_t[::1] topics
    cdef np_uint_t i
    topics = zeros((num_docs,), dtype=np_long)
    for i in xrange(num_docs):
        topics[i] = uint_argmax(dt_counts[i, :], num_topics)
    return topics

cdef void eval_pf(np_uint_t num_topics, ParticleFilter pf,
        list test_sample, list test_labels, list train_labels,
        np_uint_t test_num_iters, list categories):
    cdef FirstMomentPLFilter plfilter
    cdef GlobalModel model
    cdef GibbsSampler gibbs_sampler
    cdef np_double_t ll
    cdef np_long_t[::1] inferred_topics

    model = pf.max_posterior_model()

    inferred_topics = pf.label_store.label_view(pf.max_posterior_particle())
    print('in-sample nmi: %f'
        % nmi(train_labels, categories, inferred_topics, num_topics))

    gibbs_sampler = GibbsSampler(model)
    gibbs_sampler.infer(test_sample, test_num_iters)
    inferred_topics = infer_topics(gibbs_sampler.dt_counts, len(test_sample),
        num_topics)
    print('out-of-sample nmi: %f'
        % nmi(test_labels, categories, inferred_topics, num_topics))

    plfilter = FirstMomentPLFilter(model)
    ll = plfilter.likelihood(test_sample)
    print('out-of-sample log-likelihood: %f' % ll)
    print('out-of-sample perplexity: %f' % perplexity(ll, test_sample))


def create_pf(GlobalModel model, list init_sample,
        np_uint_t[:, ::1] dt_counts, np_uint_t[::1] d_counts,
        list assignments, dict params):
    cdef ParticleFilter pf
    cdef ReservoirSampler rs
    cdef ParticleFilterReservoirData rejuv_data
    cdef ParticleLabelStore label_store
    cdef bint ejected, inserted
    cdef lowl_key ejected_token_idx
    cdef size_t reservoir_idx
    cdef np_uint_t ret, token_idx
    cdef np_uint_t[::1] particle_d_counts, zz
    cdef np_uint_t[:, ::1] particle_dt_counts

    label_store = ParticleLabelStore(params['num_particles'],
        params['num_topics'])
    rejuv_data = ParticleFilterReservoirData(params['reservoir_size'],
        params['num_particles'], params['num_topics'])
    rs = ReservoirSampler()
    ret = rs.init(params['reservoir_size'])
    token_idx = 0

    for doc_idx in xrange(len(init_sample)):
        for j in xrange(len(init_sample[doc_idx])):
            w = init_sample[doc_idx][j]
            inserted = rs._insert(token_idx, &reservoir_idx,
                &ejected, &ejected_token_idx)
            if inserted:
                zz = zeros(
                    (params['num_particles'],), dtype=np_uint)
                particle_d_counts = zeros(
                    (params['num_particles'],), dtype=np_uint)
                particle_dt_counts = zeros(
                    (params['num_particles'], params['num_topics']),
                    dtype=np_uint)
                for p in xrange(params['num_particles']):
                    zz[p] = assignments[token_idx]
                    particle_d_counts[p] = d_counts[doc_idx]
                    particle_dt_counts[p,:] = dt_counts[doc_idx,:]
                rejuv_data.insert(reservoir_idx, doc_idx, w, zz,
                    particle_d_counts, particle_dt_counts)
            token_idx += 1

        for p in xrange(params['num_particles']):
            label_store.append(p,
                label_store.compute_label(dt_counts[doc_idx, :]))

    pf = ParticleFilter(model, params['num_particles'], params['ess_threshold'],
        rs, rejuv_data, params['rejuv_sample_size'], params['rejuv_mcmc_steps'],
        token_idx, label_store, params['resample_propagate'])
    return pf


def init_lda(list init_sample, list init_labels, list categories,
        np_uint_t vocab_size, dict params):
    cdef np_uint_t[::1] t_counts
    cdef np_uint_t[:, ::1] tw_counts
    cdef GlobalModel orig_model, best_model, model
    cdef FirstMomentPLFilter plfilter
    cdef GibbsSampler init_gibbs_sampler, best_init_gibbs_sampler, eval_gs
    cdef ParticleFilter pf
    cdef list init_train_sample, init_eval_sample
    cdef list init_train_labels, init_eval_labels
    cdef list best_init_train_sample, best_init_train_labels
    cdef list limits
    cdef np_uint_t i, j, k, r, b, init_train_size, num_tokens, base_fold_size
    cdef np_double_t score, best_score
    cdef np_long_t[::1] inferred_topics
    cdef list scores, models, init_gibbs_samplers, init_train_sample_lists
    cdef list init_train_label_lists

    tw_counts = zeros((params['num_topics'], vocab_size), dtype=np_uint)
    t_counts = zeros((params['num_topics'],), dtype=np_uint)
    orig_model = GlobalModel(tw_counts, t_counts, params['alpha'],
        params['beta'], params['num_topics'], vocab_size)

    reseed = None
    if params['init_tune_seed'] >= 0:
        print('fixing prng seed to %d for initialization'
            % params['init_tune_seed'])
        reseed = randint(0, 1e9)
        seed(params['init_tune_seed'])

    if params['init_tune_num_runs'] > 1:
        if params['init_tune_num_cv_folds'] == 0:
            if not params['init_tune_eval_nmi']:
                print('warning: in-sample likelihood is not supported')
            print('initializing from best run of %d, by in-sample nmi'
                % params['init_tune_num_runs'])
            for i in xrange(params['init_tune_num_runs']):
                model = orig_model.copy()
                init_gibbs_sampler = GibbsSampler(model)
                init_gibbs_sampler.learn(init_sample, params['init_num_iters'])
                inferred_topics = zeros((len(init_sample),), dtype=np_long)
                for j in xrange(len(init_sample)):
                    inferred_topics[j] = uint_argmax(
                        init_gibbs_sampler.dt_counts[j,:], params['num_topics'])
                score = nmi(init_labels, categories, inferred_topics,
                    params['num_topics'])
                print('result: %f' % score)

                if i == 0 or score > best_score:
                    best_score = score
                    best_model = model
                    best_init_gibbs_sampler = init_gibbs_sampler
                    best_init_train_sample = init_sample
                    best_init_train_labels = init_labels

            print('best run result: %f' % best_score)

        elif params['init_tune_num_cv_folds'] == 1:
            init_train_size = int(len(init_sample)
                * params['init_tune_train_frac'])

            if params['init_tune_eval_nmi']:
                print('initializing from best run of %d, by out-of-sample nmi'
                    % params['init_tune_num_runs'])
            else:
                print('initializing from best run of %d, by out-of-sample ll'
                    % params['init_tune_num_runs'])

            print('training on %f%% (%d docs), evaluating on remainder'
                % (100 * params['init_tune_train_frac'], init_train_size))

            init_eval_sample = init_sample[init_train_size:]
            init_eval_labels = init_labels[init_train_size:]
            init_train_sample = init_sample[:init_train_size]
            init_train_labels = init_labels[:init_train_size]

            for i in xrange(params['init_tune_num_runs']):
                model = orig_model.copy()
                init_gibbs_sampler = GibbsSampler(model)
                init_gibbs_sampler.learn(init_train_sample,
                    params['init_num_iters'])

                if params['init_tune_eval_nmi']:
                    eval_gs = GibbsSampler(model)
                    eval_gs.infer(init_eval_sample,
                        params['test_num_iters'])
                    inferred_topics = infer_topics(eval_gs.dt_counts,
                        len(init_eval_sample), params['num_topics'])
                    score = nmi(init_eval_labels, categories,
                        inferred_topics, params['num_topics'])
                else:
                    plfilter = FirstMomentPLFilter(model)
                    num_tokens = 0
                    for k in xrange(len(init_eval_sample)):
                        num_tokens += len(init_eval_sample[k])
                    score = (plfilter.likelihood(init_eval_sample)
                        / num_tokens)

                print('result: %f' % score)

                if i == 0 or score > best_score:
                    best_score = score
                    best_model = model
                    best_init_gibbs_sampler = init_gibbs_sampler
                    best_init_train_sample = init_train_sample
                    best_init_train_labels = init_train_labels

            print('best run result: %f' % best_score)

        else:
            limits = [0]
            base_fold_size = len(init_sample) / params['init_tune_num_cv_folds']
            r = (len(init_sample)
                - base_fold_size * params['init_tune_num_cv_folds'])
            for i in xrange(params['init_tune_num_cv_folds']):
                b = limits[-1] + base_fold_size
                if i < r:
                    b += 1
                limits.append(b)

            if params['init_tune_eval_nmi']:
                print('initializing from best run of %d, by out-of-sample nmi'
                    % params['init_tune_num_runs'])
            else:
                print('initializing from best run of %d, by out-of-sample ll'
                    % params['init_tune_num_runs'])

            print('using cross-validation with %d folds with sizes:'
                % params['init_tune_num_cv_folds'])
            for j in xrange(params['init_tune_num_cv_folds']):
                sys.stdout.write('  %d' % (limits[j+1] - limits[j]))
            sys.stdout.write('\n')
            sys.stdout.flush()

            for i in xrange(params['init_tune_num_runs']):
                scores = []
                models = []
                init_gibbs_samplers = []
                init_train_sample_lists = []
                init_train_label_lists = []

                for j in xrange(params['init_tune_num_cv_folds']):
                    init_eval_sample = init_sample[limits[j]:limits[j+1]]
                    init_eval_labels = init_labels[limits[j]:limits[j+1]]
                    init_train_sample = (init_sample[:limits[j]]
                        + init_sample[limits[j+1]:])
                    init_train_labels = (init_labels[:limits[j]]
                        + init_labels[limits[j+1]:])

                    model = orig_model.copy()
                    init_gibbs_sampler = GibbsSampler(model)
                    init_gibbs_sampler.learn(init_train_sample,
                        params['init_num_iters'])

                    if params['init_tune_eval_nmi']:
                        eval_gs = GibbsSampler(model)
                        eval_gs.infer(init_eval_sample,
                            params['test_num_iters'])
                        inferred_topics = infer_topics(eval_gs.dt_counts,
                            len(init_eval_sample), params['num_topics'])
                        score = nmi(init_eval_labels, categories,
                            inferred_topics, params['num_topics'])
                    else:
                        plfilter = FirstMomentPLFilter(model)
                        num_tokens = 0
                        for k in xrange(len(init_eval_sample)):
                            num_tokens += len(init_eval_sample[k])
                        score = (plfilter.likelihood(init_eval_sample)
                            / num_tokens)

                    scores.append(score)
                    models.append(model)
                    init_gibbs_samplers.append(init_gibbs_sampler)
                    init_train_sample_lists.append(init_train_sample)
                    init_train_label_lists.append(init_train_labels)

                print('cross-validation results:')
                for j in xrange(params['init_tune_num_cv_folds']):
                    sys.stdout.write('  %f' % scores[j])
                sys.stdout.write('\n')
                sys.stdout.flush()

                idx_score_pairs = zip(range(params['init_tune_num_cv_folds']),
                    scores)
                idx_score_pairs.sort(key=lambda p: p[1])
                m = idx_score_pairs[len(idx_score_pairs)/2][0]
                print('median result: %f' % scores[m])

                if i == 0 or scores[m] > best_score:
                    best_score = scores[m]
                    best_model = models[m]
                    best_init_gibbs_sampler = init_gibbs_samplers[m]
                    best_init_train_sample = init_train_sample_lists[m]
                    best_init_train_labels = init_train_label_lists[m]

            print('best run (median) result: %f' % best_score)

    else:
        print('gibbs sampling with %d iters' % params['init_num_iters'])
        best_model = orig_model
        best_init_gibbs_sampler = GibbsSampler(best_model)
        best_init_gibbs_sampler.learn(init_sample, params['init_num_iters'])
        best_init_train_sample = init_sample
        best_init_train_labels = init_labels

    print('creating particle filter on initialized model')
    pf = create_pf(best_model, best_init_train_sample,
        best_init_gibbs_sampler.dt_counts,
        best_init_gibbs_sampler.d_counts,
        best_init_gibbs_sampler.assignments,
        params)

    if reseed is not None:
        print('reseeding prng with %d' % reseed)
        seed(reseed)

    num_tokens = 0
    for i in xrange(len(best_init_train_sample)):
        num_tokens += len(best_init_train_sample[i])

    return (pf, best_init_train_labels, len(best_init_train_sample), num_tokens)


def run_lda(data_dir, categories, **kwargs):
    cdef ParticleFilter pf
    cdef np_uint_t i, doc_idx, num_tokens, p

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

    print('vocab size: %d' % len(dataset.vocab))

    def preprocess(doc_triple):
        # replace words with unique ids
        return doc_triple[:2] + ([dataset.vocab[w] for w in doc_triple[2]],)

    test_data = [preprocess(t) for t in dataset.test_iterator()]
    test_sample = [t[2] for t in test_data]
    test_labels = [t[1] for t in test_data]

    init_sample = []
    init_labels = []

    pf = None

    i = 0

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
                (pf, train_labels, doc_idx, num_tokens) = init_lda(
                    init_sample, init_labels, list(categories),
                    len(dataset.vocab), params)

                eval_pf(params['num_topics'], pf,
                    test_sample, test_labels, train_labels,
                    params['test_num_iters'], list(categories))
                print(pf.max_posterior_model().to_string(dataset.vocab, 20))

            # process current document through pf
            pf.step(doc_idx, d[2])
            train_labels.append(d[1])
            print('doc: %d' % doc_idx)
            print('num words: %d' % len(d[2]))
            if doc_idx % 50 == 0:
                eval_pf(params['num_topics'], pf,
                    test_sample, test_labels, train_labels,
                    params['test_num_iters'], list(categories))

            doc_idx += 1
            num_tokens += len(d[2])

        i += 1

    if i <= params['init_num_docs']:
        # init_num_docs was really big; do Gibbs sampling and initialize
        # pf just so we can evaluate the model learned by Gibbs
        (pf, train_labels, doc_idx, num_tokens) = init_lda(
            init_sample, init_labels, list(categories),
            len(dataset.vocab), params)

    # end of run, do one last eval and print topics
    eval_pf(params['num_topics'], pf, test_sample, test_labels,
        train_labels, params['test_num_iters'], list(categories))

    print('trained on %d docs (%d tokens)' % (doc_idx, num_tokens))
    print(pf.max_posterior_model().to_string(dataset.vocab, 20))
