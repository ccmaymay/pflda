#!/usr/bin/env python

from random import random, randint, seed
from data import Dataset, TopicList
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

    # size of reservoir, in tokens, used by particle filter
    reservoir_size = 1000,

    # number of gibbs iterations (one per token) used when computing
    # out-of-sample nmi
    test_num_iters = 5,

    # dirichlet hyperparameter on topic distributions
    alpha = 0.1,

    # dirichlet hyperparameter on word distributions
    beta = 0.1,

    # threshold for effective sample size over particle weights:
    # if computed ess drops below this
    # value, the particles will be resampled
    ess_threshold = 20.0,

    # whether to check ess and (conditionally) resample and
    # rejuvenate after every token, or only at the end of each
    # document
    resample_per_token = False,

    # whether to subtract counts from the word distributions when
    # a document is ejected from the reservoir (or not inserted)
    forget_stats = False,

    # total number of docs to use to initialize model (via gibbs
    # sampling) for particle filter
    init_num_docs = 100,

    # number of gibbs sweeps to use when initializing model
    init_num_iters = 100,

    # prng seed for initialization: if non-negative, prng will be seeded
    # with this value before initialization and re-seeded randomly
    # afterward; if negative, do not seed prng (python chooses a random
    # state for us at start-up)
    init_seed = -1,

    # number of runs to perform when initializing model: if 0 or 1,
    # initialize model using init_num_iters gibbs sweeps over the
    # initialization sample; if greater than 1, initialize this many
    # models (using init_num_iters gibbs sweeps over the initialization
    # sample each) and create the particle filter from the
    # highest-scoring model
    init_tune_num_runs = 1,

    # when init_tune_num_runs is greater than 1, this boolean controls
    # whether we use NMI (True) or perplexity (False) to score
    # the models
    init_tune_eval_nmi = False,

    # when init_tune_num_runs is greater than 1, this controls whether
    # we choose our initialization model by in-sample evaluation (0),
    # simple out-of-sample evaluation (1), or cross-validation
    # (any value greater than one---in which case this is the number
    # of folds used)
    init_tune_num_cv_folds = 1,

    # when init_tune_num_runs is greater than 1 and
    # init_tune_num_cv_folds is 1, during initialization, train each
    # model on this fraction of the initialization data
    # (that is, fraction of init_num_docs) and evaluate on the rest
    # (the split will be the same for all runs)
    init_tune_train_frac = 0.8,

    # number of particles used in particle filter
    num_particles = 100,

    # size of sample drawn from reservoir for rejuvenation, in tokens
    rejuv_sample_size = 30,

    # number of gibbs sweeps performed over sample during rejuvenation
    rejuv_mcmc_steps = 1,

    # number of topics
    num_topics = 3,

    # number of words to use in coherence estimation
    coherence_num_words = 10,

    # number of particles used in left-to-right likelihood estimation
    ltr_num_particles = 20,

    # controls whether we use the left-to-right algorithm to perform
    # likelihood estimation, in addition to the first moment PL
    # approximation; note that this may increase runtime significantly
    ltr_eval = False,

    # path to file containing initial topic distributions; the
    # counts in this file are *added* to the counts initialized
    # by the gibbs sampler or particle filter, so this is a bit of
    # a hack meant to be used mainly for diagnostics.
    init_topic_list_filename = '',
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
    cdef np_uint_t n, i, j, temp
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
    cdef np_uint_t n, i, j, temp

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


# compute entropy of a list of (document) labels, given list of
# available label types
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


# compute entropy of a list of (document) topics, given number of
# available topics (topic values are assumed to be in [0, num_topics))
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


# compute mutual information between given document labels and
# inferred topics (assumed to be within [0, num_topics))
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


# compute normalized mutual information between given document labels
# and inferred topics (assumed to be within [0, num_topics))
cdef np_double_t nmi(list labels, list label_types,
        np_long_t[::1] inferred_topics, np_uint_t num_topics):
    cdef np_double_t _nmi

    _nmi = 2.0 * (mi(labels, label_types, inferred_topics, num_topics) /
        (entropy1(labels, label_types) + entropy2(inferred_topics, num_topics)))

    return _nmi


# representation of labels inferred by particle filter for a (growing)
# sequence of docs (one label per doc per particle); labels are assumed
# to be in [0, num_topics) (integers)
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

    cdef void set(self, np_uint_t p, np_uint_t doc_idx, np_uint_t label):
        self.labels[p][doc_idx] = label

    cdef long compute_label(self, np_uint_t[::1] dt_counts):
        return uint_argmax(dt_counts)

    cdef void recompute(self, ParticleFilterReservoirData rejuv_data):
        cdef np_uint_t[::1] dt_counts
        cdef np_uint_t p, reservoir_idx, doc_idx
        cdef long label

        for p in xrange(self.num_particles):
            for reservoir_idx in xrange(rejuv_data.occupied):
                doc_idx = rejuv_data.doc_ids[reservoir_idx]
                dt_counts = rejuv_data.dt_counts[reservoir_idx, p, :]
                label = self.compute_label(dt_counts)
                self.set(p, doc_idx, label)

    cdef void copy_particle(self, np_uint_t old_p, np_uint_t new_p):
        cdef np_uint_t i
        for i in xrange(len(self.labels[old_p])):
            self.labels[new_p][i] = self.labels[old_p][i]

    cdef np_long_t[::1] label_view(self, np_uint_t p):
        cdef list particle_labels
        cdef np_long_t[::1] view
        cdef np_uint_t i
        particle_labels = self.labels[p]
        view = zeros((len(particle_labels),), dtype=np_long)
        for i in xrange(len(particle_labels)):
            view[i] = particle_labels[i]
        return view


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
        

# estimator of document coherence as defined in Mimno et al. (2011)
cdef class CoherenceEstimator:
    cdef GlobalModel model
    cdef np_uint_t num_words

    def __cinit__(self, GlobalModel model, np_uint_t num_words):
        self.model = model
        self.num_words = num_words

    cdef np_double_t coherence(self, list sample):
        cdef np_uint_t t, w, i, j, doc_idx, word_idx, doc_freq, joint_doc_freq
        cdef np_double_t avg
        cdef np_uint_t[::1] w_counts, w_indices, r_w_indices
        cdef np_uint_t[:, ::1] sample_w_counts
        cdef list doc

        w_counts = zeros((self.model.vocab_size,), dtype=np_uint)
        r_w_indices = zeros((self.model.vocab_size,), dtype=np_uint)
        sample_w_counts = zeros((len(sample), self.num_words), dtype=np_uint)

        avg = 0.0

        for t in xrange(self.model.num_topics):
            w_counts[:] = self.model.tw_counts[t,:]
            w_indices = reverse_sort_uint(w_counts, self.num_words)

            for w in xrange(self.model.vocab_size):
                r_w_indices[w] = self.num_words
            for i in xrange(self.num_words):
                r_w_indices[w_indices[i]] = i

            for doc_idx in xrange(len(sample)):
                for i in xrange(self.num_words):
                    sample_w_counts[doc_idx, i] = 0
                doc = sample[doc_idx]
                for word_idx in xrange(len(doc)):
                    w = doc[word_idx]
                    i = r_w_indices[w]
                    if i < self.num_words:
                        sample_w_counts[doc_idx, i] += 1

            for i in xrange(1, self.num_words):
                for j in xrange(i):
                    doc_freq = 0
                    joint_doc_freq = 0
                    for doc_idx in xrange(len(sample)):
                        if sample_w_counts[doc_idx, j] > 0:
                            doc_freq += 1
                            if sample_w_counts[doc_idx, i] > 0:
                                joint_doc_freq += 1
                    avg += log(joint_doc_freq + 1.0) - log(doc_freq)

        avg /= self.model.num_topics

        return avg


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


# compact, dense representation of docs stored in reservoir and
# particle filter state corresponding to those docs
cdef class ParticleFilterReservoirData:
    # map from reservoir idx -> doc idx
    cdef np_uint_t[::1] doc_ids
    # map from reservoir idx -> word id list
    cdef list w
    # map from reservoir idx -> particle -> topic assignment list
    cdef list z

    # map from (reservoir idx, particle) -> document count
    cdef np_uint_t[:, ::1] d_counts
    # map from (reservoir idx, particle, topic) ->
    # document-topic count
    cdef np_uint_t[:, :, ::1] dt_counts

    cdef np_uint_t capacity, occupied, num_particles, num_topics

    def __cinit__(self, np_uint_t capacity, np_uint_t num_particles,
            np_uint_t num_topics):
        self.doc_ids = zeros((capacity,), dtype=np_uint)
        self.w = list()
        self.z = list()
        self.d_counts = zeros((capacity, num_particles), dtype=np_uint)
        self.dt_counts = zeros(
            (capacity, num_particles, num_topics), dtype=np_uint)

        self.capacity = capacity
        self.num_particles = num_particles
        self.num_topics = num_topics
        self.occupied = 0

    # update corresponding to reservoir insert
    cdef void insert(self, np_uint_t reservoir_idx,
            np_uint_t doc_idx, list _w):
        cdef np_uint_t i
        if reservoir_idx < self.occupied:
            self.w[reservoir_idx] = _w
            self.z[reservoir_idx] = list()
        else:
            self.w.append(_w)
            self.z.append(list())
            self.occupied += 1
        for i in xrange(self.num_particles):
            self.z[reservoir_idx].append(list())
        self.d_counts[reservoir_idx,:] = 0
        self.dt_counts[reservoir_idx,:,:] = 0
        self.doc_ids[reservoir_idx] = doc_idx

    cdef void transition_z(self, np_uint_t reservoir_idx, np_uint_t p,
            np_uint_t _z):
        self.z[reservoir_idx][p].append(_z)

    cdef void copy_particle(self, np_uint_t old_p, np_uint_t new_p):
        cdef np_uint_t i, j
        self.d_counts[:, new_p] = self.d_counts[:, old_p]
        self.dt_counts[:, new_p, :] = self.dt_counts[:, old_p, :]
        for i in xrange(self.occupied):
            for j in xrange(len(self.z[i][old_p])):
                self.z[i][new_p][j] = self.z[i][old_p][j]

    def to_string(self):
        cdef np_uint_t i, j, p, k
        cdef list doc, zz
        cdef str s

        s = ''

        s += 'rd_doc_ids:'
        for i in xrange(self.occupied):
            s += ' %d' % self.doc_ids[i]

        s += '\n'

        s += 'rd_w:\n'
        for i in xrange(self.occupied):
            s += ' '
            doc = self.w[i]
            for j in xrange(len(doc)):
                s += ' %d' % doc[j]
            s += '\n'

        s += 'rd_z:\n'
        for i in xrange(self.occupied):
            for p in xrange(self.num_particles):
                zz = self.z[i][p]
                s += ' '
                for j in xrange(len(zz)):
                    s += ' %d' % zz[j]
                s += '\n'
            s += '\n'

        s += 'rd_dt_counts:\n'
        for i in xrange(self.occupied):
            s += ' '
            for j in xrange(self.num_particles):
                for k in xrange(self.num_topics):
                    s += ' %d' % self.dt_counts[i, j, k]
                if j + 1 < self.num_particles:
                    s += ','
            s += '\n'

        return s


# particle filter for LDA, with rejuvenation sequence based on reservoir
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
    cdef np_uint_t num_topics, vocab_size, num_particles
    cdef np_uint_t rejuv_sample_size, rejuv_mcmc_steps
    cdef np_double_t alpha, beta, ess_threshold
    cdef bint debug, resample_per_token, forget_stats

    def __cinit__(self, GlobalModel init_model, np_uint_t num_particles,
            np_double_t ess_threshold, ReservoirSampler rs,
            ParticleFilterReservoirData rejuv_data,
            np_uint_t rejuv_sample_size, np_uint_t rejuv_mcmc_steps,
            ParticleLabelStore label_store, bint resample_per_token,
            bint forget_stats, bint debug):
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

        self.resample_per_token = resample_per_token
        self.forget_stats = forget_stats
        self.debug = debug

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

        if self.debug:
            print('resampling map:')

        ids_to_resample = zeros((self.num_particles,), dtype=np_uint)
        filled_slots = zeros((self.num_particles,), dtype=np_uint)
        for i in xrange(self.num_particles):
            j = self.sample_particle_num()
            if filled_slots[j] == 0:
                filled_slots[j] = 1
                if self.debug:
                    print('  %d -> %d' % (j, j))
            else:
                ids_to_resample[j] += 1

        for i in xrange(self.num_particles):
            while ids_to_resample[i] > 0:
                j = 0
                while filled_slots[j] == 1:
                    j += 1
                if self.debug:
                    print('  %d -> %d' % (j, i))
                self.tw_counts[j, :, :] = self.tw_counts[i, :, :]
                self.t_counts[j, :] = self.t_counts[i, :]
                self.rejuv_data.copy_particle(i, j)
                self.label_store.copy_particle(i, j)
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
        cdef lowl_key ejected_doc_idx
        cdef size_t reservoir_idx
        cdef np_uint_t z, i, t, num_tokens
        cdef np_uint_t[:, ::1] zz
        cdef np_double_t total_weight, prior, _ess
        cdef bint inserted, ejected

        num_tokens = len(doc)
        zz = zeros((self.num_particles, num_tokens), dtype=np_uint)

        inserted = self.rs._insert(doc_idx, &reservoir_idx,
            &ejected, &ejected_doc_idx)
        if inserted:
            if self.debug:
                print(self.rejuv_data.to_string())
                if ejected:
                    print('rsvr replace: doc_idx %d -> %d; reservoir_idx %d'
                        % (ejected_doc_idx, doc_idx, reservoir_idx))
                else:
                    print('rsvr insert: doc_idx %d; reservoir_idx %d'
                        % (doc_idx, reservoir_idx))

            if ejected and self.forget_stats:
                for j in xrange(len(self.rejuv_data.w[reservoir_idx])):
                    w = self.rejuv_data.w[reservoir_idx][j]
                    for i in xrange(self.num_particles):
                        z = self.rejuv_data.z[reservoir_idx][i][j]
                        self.tw_counts[i, z, w] -= 1
                        self.t_counts[i, z] -= 1

            self.rejuv_data.insert(reservoir_idx, doc_idx, doc)

            # set local_d_counts and local_dt_counts to point to
            # the unique set of sufficient statistics for this
            # document in the reservoir data; the rest of the
            # particle filter steps performed for this document
            # will update these arrays directly
            self.local_d_counts = self.rejuv_data.d_counts[reservoir_idx,:]
            self.local_dt_counts = self.rejuv_data.dt_counts[reservoir_idx,:,:]

            if self.debug:
                print(self.rejuv_data.to_string())
        else:
            # initialize document sufficient statistics to zero.  note that
            # if we add a token from this document to the reservoir, these
            # members will be changed to point to the (unique) set of
            # document sufficient statistics in the reservoir data
            self.local_d_counts = zeros(
                (self.num_particles,), dtype=np_uint)
            self.local_dt_counts = zeros(
                (self.num_particles, self.num_topics), dtype=np_uint)

        for i in xrange(self.num_particles):
            self.label_store.append(i, 0)

        for j in xrange(num_tokens):
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
            if self.debug:
                sys.stdout.write('weights:')
                for i in xrange(self.num_particles):
                    sys.stdout.write(' %f' % self.weights[i])
                sys.stdout.write('\n')
                sys.stdout.flush()

            for i in xrange(self.num_particles):
                z = self.sample_topic(i, w)
                zz[i, j] = z
                if inserted:
                    self.rejuv_data.transition_z(reservoir_idx, i, z)
                self.tw_counts[i, z, w] += 1
                self.t_counts[i, z] += 1
                self.local_dt_counts[i, z] += 1
                self.local_d_counts[i] += 1

            _ess = self.ess()
            if self.resample_per_token and (_ess < self.ess_threshold):
                if self.debug:
                    print(self.rejuv_data.to_string())
                print('resampling: ess %f; doc_idx %d, %d' % (_ess, doc_idx, j))
                self.resample()
                if self.debug:
                    print(self.rejuv_data.to_string())
                self.rejuvenate()
                self.label_store.recompute(self.rejuv_data)
                if self.debug:
                    print(self.rejuv_data.to_string())

            PyErr_CheckSignals()

        _ess = self.ess()
        if (not self.resample_per_token) and (_ess < self.ess_threshold):
            if self.debug:
                print(self.rejuv_data.to_string())
            print('resampling: ess %f; doc_idx %d, %d' % (_ess, doc_idx, j))
            self.resample()
            if self.debug:
                print(self.rejuv_data.to_string())
            self.rejuvenate()
            self.label_store.recompute(self.rejuv_data)
            if self.debug:
                print(self.rejuv_data.to_string())

        if self.forget_stats and not inserted:
            for j in xrange(num_tokens):
                w = doc[j]
                for i in xrange(self.num_particles):
                    z = zz[i, j] = z
                    self.tw_counts[i, z, w] -= 1
                    self.t_counts[i, z] -= 1

        for i in xrange(self.num_particles):
            self.label_store.set(i, doc_idx,
                self.label_store.compute_label(self.local_dt_counts[i, :]))

    cdef void rejuvenate(self):
        cdef GlobalModel model
        cdef np_uint_t[::1] sample
        cdef np_uint_t p, t, j, i, z, reservoir_idx, w
        cdef list doc, zz

        sample = sample_without_replacement(self.rs.occupied(),
            self.rejuv_sample_size)
        if self.debug:
            sys.stdout.write('rejuvenating:')
            for j in xrange(len(sample)):
                sys.stdout.write(' %d' % sample[j])
            sys.stdout.write('\n')
            sys.stdout.flush()

        for p in xrange(self.num_particles):
            for t in xrange(self.rejuv_mcmc_steps):
                for i in xrange(len(sample)):
                    reservoir_idx = sample[i]
                    doc = self.rejuv_data.w[reservoir_idx]
                    zz = self.rejuv_data.z[reservoir_idx][p]
                    for j in xrange(len(zz)):
                        z = zz[j]
                        w = doc[j]

                        self.tw_counts[p, z, w] -= 1
                        self.t_counts[p, z] -= 1
                        self.rejuv_data.dt_counts[reservoir_idx, p, z] -= 1
                        self.rejuv_data.d_counts[reservoir_idx, p] -= 1

                        z = sample_topic(
                            self.tw_counts[p, :, :], self.t_counts[p, :],
                            self.rejuv_data.dt_counts[reservoir_idx, p, :],
                            self.rejuv_data.d_counts[reservoir_idx, p],
                            self.alpha, self.beta,
                            self.vocab_size, self.num_topics,
                            w, self.pmf)
                        zz[j] = z

                        self.tw_counts[p, z, w] += 1
                        self.t_counts[p, z] += 1
                        self.rejuv_data.dt_counts[reservoir_idx, p, z] += 1
                        self.rejuv_data.d_counts[reservoir_idx, p] += 1

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
        p = udouble_argmax(self.weights)
        return p

    cdef GlobalModel max_posterior_model(self):
        cdef GlobalModel model
        cdef np_uint_t p
        p = self.max_posterior_particle()
        model = self.model_for_particle(p)
        return model


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


# given array of document-topic counts, compute array of inferred topics
# (one per document): the inferred topic for a document is the one that
# has been assigned to the most tokens in that document
cdef np_long_t[::1] infer_topics(np_uint_t[:, ::1] dt_counts,
        np_uint_t num_docs, np_uint_t num_topics):
    cdef np_long_t[::1] topics
    cdef np_uint_t i
    topics = zeros((num_docs,), dtype=np_long)
    for i in xrange(num_docs):
        topics[i] = uint_argmax(dt_counts[i, :])
    return topics


# evaluate LDA model (learned by Gibbs sampler) according to a
# number of metrics, in-sample and out-of-sample, printing results
# to standard output
cdef void eval_gibbs(np_uint_t num_topics, GibbsSampler train_gibbs_sampler,
        list test_sample, list test_labels, list train_labels,
        np_uint_t test_num_iters, list categories,
        np_uint_t coherence_num_words, bint ltr_eval,
        np_uint_t ltr_num_particles, np_uint_t init_size):
    cdef FirstMomentPLFilter plfilter
    cdef LeftToRightEvaluator ltr_evaluator
    cdef CoherenceEstimator coherence_est
    cdef GlobalModel model
    cdef GibbsSampler gibbs_sampler
    cdef np_double_t ll
    cdef np_long_t[::1] inferred_topics

    model = train_gibbs_sampler.model

    inferred_topics = infer_topics(train_gibbs_sampler.dt_counts,
        len(train_labels), num_topics)
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

    if ltr_eval:
        ltr_evaluator = LeftToRightEvaluator(model, ltr_num_particles)
        ll = ltr_evaluator.likelihood(test_sample)
        print('out-of-sample ltr log-likelihood: %f' % ll)
        print('out-of-sample ltr perplexity: %f' % perplexity(ll, test_sample))

    coherence_est = CoherenceEstimator(model, coherence_num_words)
    print('out-of-sample coherence: %f' % coherence_est.coherence(test_sample))


# evaluate particle filter according to a number of metrics, in-sample
# and out-of-sample, printing results to standard output
cdef void eval_pf(np_uint_t num_topics, ParticleFilter pf,
        list test_sample, list test_labels, list train_labels,
        np_uint_t test_num_iters, list categories,
        np_uint_t coherence_num_words, bint ltr_eval,
        np_uint_t ltr_num_particles, np_uint_t init_size):
    cdef FirstMomentPLFilter plfilter
    cdef LeftToRightEvaluator ltr_evaluator
    cdef CoherenceEstimator coherence_est
    cdef GlobalModel model
    cdef GibbsSampler gibbs_sampler
    cdef np_double_t ll
    cdef np_long_t[::1] inferred_topics

    model = pf.max_posterior_model()

    inferred_topics = pf.label_store.label_view(pf.max_posterior_particle())
    print('init in-sample nmi: %f'
        % nmi(train_labels[:init_size], categories, inferred_topics[:init_size],            num_topics))

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

    if ltr_eval:
        ltr_evaluator = LeftToRightEvaluator(model, ltr_num_particles)
        ll = ltr_evaluator.likelihood(test_sample)
        print('out-of-sample ltr log-likelihood: %f' % ll)
        print('out-of-sample ltr perplexity: %f' % perplexity(ll, test_sample))

    coherence_est = CoherenceEstimator(model, coherence_num_words)
    print('out-of-sample coherence: %f' % coherence_est.coherence(test_sample))


# create particle filter from LDA model (e.g., one initialized by gibbs
# sampling on a subset of the data)
def create_pf(GlobalModel model, list init_sample,
        np_uint_t[:, ::1] dt_counts, np_uint_t[::1] d_counts,
        list assignments, dict params):
    cdef ParticleFilter pf
    cdef ReservoirSampler rs
    cdef ParticleFilterReservoirData rejuv_data
    cdef ParticleLabelStore label_store
    cdef bint ejected, inserted
    cdef lowl_key ejected_doc_idx
    cdef size_t reservoir_idx
    cdef np_uint_t ret, doc_idx, j, w, p, token_idx, z, num_tokens

    label_store = ParticleLabelStore(params['num_particles'],
        params['num_topics'])
    rejuv_data = ParticleFilterReservoirData(params['reservoir_size'],
        params['num_particles'], params['num_topics'])
    rs = ReservoirSampler()
    ret = rs.init(params['reservoir_size'])

    token_idx = 0

    for doc_idx in xrange(len(init_sample)):
        doc = init_sample[doc_idx]
        num_tokens = len(doc)
        inserted = rs._insert(doc_idx, &reservoir_idx,
            &ejected, &ejected_doc_idx)
        if inserted:
            if ejected and params['forget_stats']:
                for j in xrange(len(rejuv_data.w[reservoir_idx])):
                    w = rejuv_data.w[reservoir_idx][j]
                    z = rejuv_data.z[reservoir_idx][0][j]
                    model.tw_counts[z, w] -= 1
                    model.t_counts[z] -= 1
            rejuv_data.insert(reservoir_idx, doc_idx, doc)
            for j in xrange(num_tokens):
                for p in xrange(params['num_particles']):
                    z = assignments[token_idx]
                    rejuv_data.transition_z(reservoir_idx, p, z)
                token_idx += 1
        else:
            token_idx += num_tokens
        for p in xrange(params['num_particles']):
            label_store.append(p,
                label_store.compute_label(dt_counts[doc_idx, :]))

    pf = ParticleFilter(model, params['num_particles'], params['ess_threshold'],
        rs, rejuv_data, params['rejuv_sample_size'], params['rejuv_mcmc_steps'],
        label_store, params['resample_per_token'], params['forget_stats'],
        params['debug'])
    return pf


# initialize LDA model on a given set of documents, using collapsed
# Gibbs sampling, and return a four-tuple: a particle filter based
# on the initialized model, the gold-standard labels of the
# initialization set, the number of documents in the initialization
# set, and the number of tokens in the initialization set.
#
# depending on the parameters passed to this method, initialization
# may be performed by simple collapsed Gibbs sampling on the entire
# sample, or after tuning (by in-sample evaluation, held-out
# evaluation, or cross-validation)
def init_lda(list init_sample, list init_labels, list categories,
        dict vocab, dict params):
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
    cdef np_uint_t i, j, r, b, init_train_size, num_tokens, base_fold_size
    cdef np_uint_t vocab_size
    cdef np_double_t score, best_score
    cdef np_long_t[::1] inferred_topics
    cdef list scores, models, init_gibbs_samplers, init_train_sample_lists
    cdef list init_train_label_lists

    vocab_size = len(vocab)

    tw_counts = zeros((params['num_topics'], vocab_size), dtype=np_uint)
    t_counts = zeros((params['num_topics'],), dtype=np_uint)
    if params['init_topic_list_filename']:
        init_topics(tw_counts, t_counts, vocab,
            params['init_topic_list_filename'])
    orig_model = GlobalModel(tw_counts, t_counts, params['alpha'],
        params['beta'], params['num_topics'], vocab_size)

    reseed = None
    if params['init_seed'] >= 0:
        print('seed: %u' % params['init_seed'])
        reseed = randint(0, 1e9)
        seed(params['init_seed'])
    else:
        rand_seed = randint(0, 1e9)
        print('seed: %u' % rand_seed)
        seed(rand_seed)

    if params['init_tune_num_runs'] > 1:
        print('initializing from best run of %d' % params['init_tune_num_runs'])
        if params['init_tune_num_cv_folds'] == 0:
            if not params['init_tune_eval_nmi']:
                print('warning: in-sample perplexity is not supported')
            print('scoring runs by in-sample nmi')
            for i in xrange(params['init_tune_num_runs']):
                model = orig_model.copy()
                init_gibbs_sampler = GibbsSampler(model)
                init_gibbs_sampler.learn(init_sample, params['init_num_iters'])
                inferred_topics = zeros((len(init_sample),), dtype=np_long)
                for j in xrange(len(init_sample)):
                    inferred_topics[j] = uint_argmax(
                        init_gibbs_sampler.dt_counts[j,:])
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
                print('scoring runs by out-of-sample nmi')
            else:
                print('scoring runs by out-of-sample (negative) perplexity')

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
                    score = -perplexity(plfilter.likelihood(init_eval_sample),
                        init_eval_sample)

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
                print('scoring runs by out-of-sample nmi')
            else:
                print('scoring runs by out-of-sample (negative) perplexity')

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
                        score = -perplexity(
                            plfilter.likelihood(init_eval_sample),
                            init_eval_sample)

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
        print('reseed: %u' % reseed)
        seed(reseed)

    num_tokens = 0
    for i in xrange(len(best_init_train_sample)):
        num_tokens += len(best_init_train_sample[i])

    return (pf, best_init_train_labels, len(best_init_train_sample), num_tokens)


def init_topics(np_uint_t[:,::1] tw_counts, np_uint_t[::1] t_counts,
        dict vocab, bytes init_topic_list_filename):
    topic_list = TopicList(init_topic_list_filename)
    for t in xrange(topic_list.num_topics()):
        topic = topic_list.topic(t)
        for (token, count) in topic.items():
            if token in vocab:
                tw_counts[t,vocab[token]] += count
                t_counts[t] += count


# driver: initialize LDA by collapsed Gibbs sampling and run particle
# filter on the rest of the data
def run_lda(data_dir, categories, **kwargs):
    cdef ParticleFilter pf
    cdef np_uint_t i, doc_idx, num_tokens, p, init_size

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

    print('categories:')
    for category in categories:
        print('\t%s' % category)

    dataset = Dataset(data_dir, set(categories), params['shuffle_data'])

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
                    dataset.vocab, params)
                init_size = len(train_labels)

                eval_pf(params['num_topics'], pf,
                    test_sample, test_labels, train_labels,
                    params['test_num_iters'], list(categories),
                    params['coherence_num_words'], params['ltr_eval'],
                    params['ltr_num_particles'], init_size)
                print(pf.max_posterior_model().to_string(dataset.vocab,
                    params['print_num_words_per_topic']))

            # process current document through pf
            print('doc: %d' % doc_idx)
            print('num words: %d' % len(d[2]))
            print('token: %d' % num_tokens)
            pf.step(doc_idx, d[2])
            train_labels.append(d[1])
            if doc_idx % 50 == 0:
                eval_pf(params['num_topics'], pf,
                    test_sample, test_labels, train_labels,
                    params['test_num_iters'], list(categories),
                    params['coherence_num_words'], params['ltr_eval'],
                    params['ltr_num_particles'], init_size)
                print(pf.max_posterior_model().to_string(dataset.vocab,
                    params['print_num_words_per_topic']))

            doc_idx += 1
            num_tokens += len(d[2])

        i += 1

    if i <= params['init_num_docs']:
        # init_num_docs was really big; do Gibbs sampling and initialize
        # pf just so we can evaluate the model learned by Gibbs
        (pf, train_labels, doc_idx, num_tokens) = init_lda(
            init_sample, init_labels, list(categories),
            dataset.vocab, params)
        init_size = len(train_labels)
        # this is a hack so we can parse the log more easily
        print('doc: %d' % doc_idx)

    # end of run, do one last eval and print topics
    eval_pf(params['num_topics'], pf, test_sample, test_labels,
        train_labels, params['test_num_iters'], list(categories),
        params['coherence_num_words'], params['ltr_eval'],
        params['ltr_num_particles'], init_size)

    print('trained on %d docs (%d tokens)' % (doc_idx, num_tokens))
    print(pf.max_posterior_model().to_string(dataset.vocab,
        params['print_num_words_per_topic']))


# driver: learn LDA by collapsed Gibbs sampling
def run_gibbs(data_dir, categories, **kwargs):
    cdef GibbsSampler gs
    cdef GlobalModel model
    cdef np_uint_t doc_idx, num_tokens, iters, i
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

    print('categories:')
    for category in categories:
        print('\t%s' % category)

    dataset = Dataset(data_dir, set(categories), params['shuffle_data'])

    print('vocab size: %d' % len(dataset.vocab))

    def preprocess(doc_triple):
        # replace words with unique ids
        return doc_triple[:2] + ([dataset.vocab[w] for w in doc_triple[2]],)

    test_data = [preprocess(t) for t in dataset.test_iterator()]
    test_sample = [t[2] for t in test_data]
    test_labels = [t[1] for t in test_data]

    train_sample = []
    train_labels = []

    doc_idx = 0
    num_tokens = 0

    for doc_triple in dataset.train_iterator():
        d = preprocess(doc_triple)
        train_sample.append(d[2])
        train_labels.append(d[1])
        doc_idx += 1
        num_tokens += len(d[2])
        if doc_idx == params['init_num_docs']:
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
    if params['init_topic_list_filename']:
        init_topics(tw_counts, t_counts, dataset.vocab,
            params['init_topic_list_filename'])
    model = GlobalModel(tw_counts, t_counts, params['alpha'],
        params['beta'], params['num_topics'], len(dataset.vocab))

    print('gibbs sampling with %d iters' % params['init_num_iters'])
    gs = GibbsSampler(model)
    gs.init(train_sample, 1)

    print('iter: 0')
    eval_gibbs(params['num_topics'], gs, test_sample, test_labels,
        train_labels, params['test_num_iters'], list(categories),
        params['coherence_num_words'], params['ltr_eval'],
        params['ltr_num_particles'], len(train_labels))
    print(model.to_string(dataset.vocab, params['print_num_words_per_topic']))

    i = 0
    while i < params['init_num_iters']:
        iters = min(params['init_num_iters'] - i, 10)
        gs.iterate(train_sample, iters, 1)

        print('iter: %d' % (i + iters))
        eval_gibbs(params['num_topics'], gs, test_sample, test_labels,
            train_labels, params['test_num_iters'], list(categories),
            params['coherence_num_words'], params['ltr_eval'],
            params['ltr_num_particles'], len(train_labels))
        print(model.to_string(dataset.vocab,
            params['print_num_words_per_topic']))

        i += iters

    print('trained on %d docs (%d tokens)' % (doc_idx, num_tokens))
    print(model.to_string(dataset.vocab, params['print_num_words_per_topic']))
