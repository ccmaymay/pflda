'''
Online variational inference for HDP with several tricks to improve
performance.  Part of code is adapted from Matt's online LDA code.
'''

import numpy as np
import scipy.special as sp
import utils
import random


MEANCHANGETHRESH = 0.00001
RANDOM_SEED = 999931111
MIN_ADDING_NOISE_POINT = 10
MIN_ADDING_NOISE_RATIO = 1
MU0 = 0.3
RHO_BOUND = 0.0


#np.random.seed(RANDOM_SEED)
#random.seed(RANDOM_SEED)


class suff_stats:

    '''
    Struct for per-document sufficient statistics for one or more
    documents.

    Members
    -------
    m_batchsize:      int, number of documents this object represents
    m_uv_ss:  (K) float, each entry of which is the sum of
                      var_phi over all tokens in the document(s)
                      allocated the corresponding (global) topic
    m_lambda_ss:    (K x Wt) float, each entry of which is the sum of
                      var_phi * zeta over all tokens of the corresponding
                      type in the document(s) that are allocated the
                      corresponding (global) topic.  Wt is the number
                      of word types spanned by the Dt document(s).
    '''

    def __init__(self, K, Wt, Dt):
        self.m_batchsize = Dt
        self.m_uv_ss = np.zeros(K)
        self.m_lambda_ss = np.zeros((K, Wt))

    def set_zero(self):
        self.m_uv_ss.fill(0.0)
        self.m_lambda_ss.fill(0.0)


class online_hdp:

    r'''
    HDP model using stick breaking.

    This class implements the following tricks not described in the
    online HDP paper:
    * optionally add noise to global topics (via ss.m_lambda_ss)
    * optionally initialize topics (lambda) using five E-steps on a
      sample of documents
    * optionally re-order topics so that more prominent topics have
      lower (earlier) indices (re-order before updating stick
      parameters)
    * do not incorporate variational expectations of log stick lengths
      until the third iteration of the E-step (and later)
    * bound rho by below, and optionally scale, so that
      \[
        \rho_t = \min{(a, b (\tau_0 + t)^{-\kappa})}
      \]
      where a is the lower bound (defined by the constant RHO_BOUND in
      the code) and b is the scaling factor (member m_scale, see below).

    Members
    -------
    m_K:                  int, first-level DP truncation
    m_T:                  int, second-level DP truncation
    m_W:                  int, size of vocabulary
    m_D:                  int, size of corpus (in documents)
    m_kappa:              float, learning rate
    m_tau:                float, slow down parameter
    m_lambda_ss:             (K x W) float, variational parameters of
                          top-level topics (lambda in paper; top-level
                          topics are phi)
    m_eta:                float, parameter on top-level topic Dirichlet
    m_alpha:              float, second-level DP concentration parameter
    m_gamma:              float, first-level DP concentration parameter
    m_uv:         (2 x K-1) float, variational parameters of
                          first-level relative stick lengths (u and v in
                          paper; rel stick lengths are beta')
    m_Elogprobw:           (K x W) float, variational expectation of
                          log word likelihoods (E log p(w | phi) in
                          paper)
    m_scale:              float, scaling factor for rho
    m_t:           int, counter of number of times we have
                          updated lambda (word likelihoods for global
                          topics) (t_0 in paper)
    m_adding_noise:       bool, true to add noise to global topics
                          (specifically, to suff stats for var params)
                          every few documents, before E-step of
                          minibatch
    m_num_docs_parsed:    int, counter of number of docs processed
                          (updated at beginning of iteration)
    m_uv_ss           (K) float, sufficient statistics for global
                          relative stick length variational parameters
                          (corresponds to suff_stats.m_uv_ss)
    m_lambda_ss_sum:         (K) float, sum of word proportions lambda
                          (per topic)
    '''

    def __init__(self, K, T, D, W, eta, alpha, gamma, kappa, tau, scale=1.0, adding_noise=False):
        self.m_W = W
        self.m_D = D
        self.m_K = K
        self.m_T = T
        self.m_alpha = alpha
        self.m_gamma = gamma

        self.m_uv = np.zeros((2, K - 1))
        self.m_uv[0] = 1.0
        #self.m_uv[1] = self.m_gamma
        # make a uniform at beginning
        # TODO why? and how is this uniform?
        self.m_uv[1] = range(K - 1, 0, -1)

        self.m_uv_ss = np.zeros(K)

        # Intuition: take 100 to be the expected document length (TODO)
        # so that there are 100D tokens in total.  Then divide that
        # count somewhat evenly (i.i.d. Gamma(1,1) distributed) between
        # each word type and topic.  *Then* subtract eta so that the
        # posterior is composed of these pseudo-counts only (maximum
        # likelihood / no prior).  (why?!  TODO)
        self.m_lambda_ss = np.random.gamma(
            1.0, 1.0, (K, W)) * D * 100 / (K * W) - eta
        self.m_eta = eta
        self.m_Elogprobw = utils.dirichlet_expectation(self.m_eta + self.m_lambda_ss)

        self.m_tau = tau + 1
        self.m_kappa = kappa
        self.m_scale = scale
        self.m_t = 0
        self.m_adding_noise = adding_noise
        self.m_num_docs_parsed = 0

        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)

    def new_init(self, c, burn_in_samples=None):
        '''
        Initialize m_lambda_ss and m_Elogprobw (and m_lambda_ss_sum) using
        five E-step trials on each of the provided documents.  (Use
        burn_in_samples of the documents, if not None, else use all
        provided documents.)
        '''

        self.m_lambda_ss = 1.0 / self.m_W + 0.01 * np.random.gamma(1.0, 1.0,
                                                                (self.m_K, self.m_W))
        self.m_Elogprobw = utils.dirichlet_expectation(self.m_eta + self.m_lambda_ss)

        if burn_in_samples is None:
            num_samples = c.num_docs
        else:
            num_samples = min(c.num_docs, burn_in_samples)
        ids = random.sample(range(c.num_docs), num_samples)
        docs = [c.docs[id] for id in ids]

        unique_words = dict()
        word_list = []
        for doc in docs:
            for w in doc.words:
                if w not in unique_words:
                    unique_words[w] = len(unique_words)
                    word_list.append(w)
        Wt = len(word_list)  # length of words in these documents

        Elogbeta = utils.expect_log_sticks(self.m_uv)  # global sticks
        for doc in docs:
            old_lambda = self.m_lambda_ss[:, word_list].copy()
            for iter in range(5):
                sstats = suff_stats(self.m_K, Wt, 1)
                doc_score = self.doc_e_step(doc, sstats, Elogbeta,
                                            unique_words, var_converge=0.0001, max_iter=5)

                self.m_lambda_ss[:, word_list] = old_lambda + \
                    sstats.m_lambda_ss / sstats.m_batchsize
                # TODO: shouldn't the parameter be eta + lambda?
                self.m_Elogprobw = utils.dirichlet_expectation(self.m_lambda_ss)

        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)

    def process_documents(self, docs, var_converge, unseen_ids=[], update=True, opt_o=True):
        '''
        Bring m_lambda_ss and m_Elogprobw up to date for the word types in
        this minibatch, do the E-step on this minibatch, optionally
        add noise to the global topics, and then do the M-step.
        Return the four-tuple (score, count, unseen_score, unseen_count)
        representing the likelihood, number of tokens, likelihood
        restricted to documents we haven't processed before, and number
        of tokens restricted to documents we haven't processed before,
        respectively.
        '''

        # Find the unique words in this mini-batch of documents...
        self.m_num_docs_parsed += len(docs)
        adding_noise = False
        adding_noise_point = MIN_ADDING_NOISE_POINT

        if self.m_adding_noise:
            if float(adding_noise_point) / len(docs) < MIN_ADDING_NOISE_RATIO:
                adding_noise_point = MIN_ADDING_NOISE_RATIO * len(docs)

            if self.m_num_docs_parsed % adding_noise_point == 0:
                adding_noise = True

        unique_words = dict()
        word_list = []
        if adding_noise:
            word_list = range(self.m_W)
            for w in word_list:
                unique_words[w] = w
        else:
            for doc in docs:
                for w in doc.words:
                    if w not in unique_words:
                        unique_words[w] = len(unique_words)
                        word_list.append(w)
        Wt = len(word_list)  # length of words in these documents

        ss = suff_stats(self.m_K, Wt, len(docs))

        Elogbeta = utils.expect_log_sticks(self.m_uv)  # global sticks

        # run variational inference on some new docs
        score = 0.0
        count = 0
        unseen_score = 0.0
        unseen_count = 0
        for i, doc in enumerate(docs):
            doc_score = self.doc_e_step(doc, ss, Elogbeta,
                                        unique_words, var_converge)
            count += doc.total
            score += doc_score
            if i in unseen_ids:
                unseen_score += doc_score
                unseen_count += doc.total

        if adding_noise:
            # add noise to the ss
            print "adding noise at this stage..."

            # old noise
            noise = np.random.gamma(1.0, 1.0, ss.m_lambda_ss.shape)
            noise_sum = np.sum(noise, axis=1)
            ratio = np.sum(ss.m_lambda_ss, axis=1) / noise_sum
            noise = noise * ratio[:, np.newaxis]

            # new noise
            #lambda_sum_tmp = self.m_W * self.m_eta + self.m_lambda_ss_sum
            #scaled_beta = 5*self.m_W * (self.m_lambda_ss + self.m_eta) / (lambda_sum_tmp[:, np.newaxis])
            #noise = np.random.gamma(shape=scaled_beta, scale=1.0)
            #noise_ratio = lambda_sum_tmp / noise_sum
            #noise = (noise * noise_ratio[:, np.newaxis] - self.m_eta) * len(docs)/self.m_D

            mu = MU0 * 1000.0 / (self.m_t + 1000)

            ss.m_lambda_ss = ss.m_lambda_ss * (1.0 - mu) + noise * mu

        if update:
            self.update_lambda(ss, opt_o, word_list)

        return (score, count, unseen_score, unseen_count)

    def optimal_ordering(self):
        '''
        Re-order global variational data-structures along dimension
        of global topics so that, roughly, the most prominent topics
        have lower indices.

        It seems that we'd like to order topics by stick lengths, in
        descending order.  The variational means are available in
        m_uv_ss.  But for some reason we order instead by the
        lambda sums, in some sense the total amount of "mass" in each
        topic's Dirichlet (again, in descending order).  This is not
        counter-intuitive but it's not clear why this is preferred over
        the stick lengths.  (TODO)
        '''

        idx = [i for i in reversed(np.argsort(self.m_lambda_ss_sum))]
        self.m_uv_ss = self.m_uv_ss[idx]
        self.m_lambda_ss = self.m_lambda_ss[idx, :]
        self.m_lambda_ss_sum = self.m_lambda_ss_sum[idx]
        self.m_Elogprobw = self.m_Elogprobw[idx, :]

    def doc_e_step(self, doc, ss, Elogbeta,
                   unique_words, var_converge,
                   max_iter=100):
        '''
        Perform document-level coordinate ascent updates of variational
        parameters.  Don't incorporate variational expectations of log
        stick lengths until the third iteration of the E-step (and
        later).  Update global sufficient statistics by incrementing
        members of ss accordingly.  Return likelihood for this E-step.

        Variables
        ---------
        v:               (2 x T-1) float, variational parameters of
                         second level relative stick lengths (a and b in
                         paper, respectively by row; rel stick lengths
                         are pi')
        zeta:             (N x T) float, variational parameters for topic
                         index of token (index into active topics for
                         this document) (zeta in paper; indices are z)
        var_phi:         (T x K) float, variational parameter for
                         local->global topic pointer (varphi in paper,
                         where topic pointer is c)
        Elogprobw_doc:    (K x N) float, expected log word likelihoods
                         (E log p(w | phi) in paper)
        Elogbeta:  (K) float, expected log first-level stick
                         lengths (E log beta in paper)
        Elogpi:  (T) float, expected log second-level stick
                         lengths (E log pi in paper)

        (denote by N the number of *types* in the document)
        '''

        batchids = [unique_words[id] for id in doc.words]

        Elogprobw_doc = self.m_Elogprobw[:, doc.words]
        # very similar to the hdp equations
        v = np.zeros((2, self.m_T - 1))
        v[0] = 1.0
        v[1] = self.m_alpha

        # The following line is of no use.
        Elogpi = utils.expect_log_sticks(v)

        # back to the uniform
        zeta = np.ones((len(doc.words), self.m_T)) * 1.0 / self.m_T

        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0
        eps = 1e-100

        iter = 0
        # not yet support second level optimization yet, to be done in the
        # future
        while iter < max_iter and (converge < 0.0 or converge > var_converge):
            # update variational parameters
            # var_phi
            if iter < 3:
                var_phi = np.dot(zeta.T,  (Elogprobw_doc * doc.counts).T)
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(
                    zeta.T,  (Elogprobw_doc * doc.counts).T) + Elogbeta
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)

            # zeta
            if iter < 3:
                zeta = np.dot(var_phi, Elogprobw_doc).T
                (log_zeta, log_norm) = utils.log_normalize(zeta)
                zeta = np.exp(log_zeta)
            else:
                zeta = np.dot(var_phi, Elogprobw_doc).T + Elogpi
                (log_zeta, log_norm) = utils.log_normalize(zeta)
                zeta = np.exp(log_zeta)

            # v
            zeta_all = zeta * np.array(doc.counts)[:, np.newaxis]
            v[0] = 1.0 + np.sum(zeta_all[:, :self.m_T - 1], 0)
            zeta_cum = np.flipud(np.sum(zeta_all[:, 1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(zeta_cum))
            Elogpi = utils.expect_log_sticks(v)

            likelihood = 0.0
            # compute likelihood
            # var_phi part
            likelihood += np.sum((Elogbeta - log_var_phi) * var_phi)

            # v part
            log_alpha = np.log(self.m_alpha)
            likelihood += (self.m_T - 1) * log_alpha
            dig_sum = sp.psi(np.sum(v, 0))
            likelihood += np.sum((np.array([1.0, self.m_alpha])
                                  [:, np.newaxis] - v) * (sp.psi(v) - dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) - \
                np.sum(sp.gammaln(v))

            # Z part
            likelihood += np.sum((Elogpi - log_zeta) * zeta)

            # X part, the data part
            likelihood += np.sum(zeta.T *
                                 np.dot(var_phi, Elogprobw_doc * doc.counts))

            converge = (likelihood - old_likelihood) / abs(old_likelihood)
            old_likelihood = likelihood

            if converge < -0.000001:
                print "warning, likelihood is decreasing!"

            iter += 1

        # update the suff_stat ss
        # this time it only contains information from one doc
        ss.m_uv_ss += np.sum(var_phi, 0)
        ss.m_lambda_ss[:, batchids] += np.dot(var_phi.T, zeta.T * doc.counts)

        return likelihood

    def update_lambda(self, sstats, opt_o, word_list):
        '''
        Perform global updates of variational parameters using the given
        sufficient statistics sstats.  (sstats represents the documents
        in the last-processed minibatch.)  Note that the global topic
        dimension in the global variational parameters (m_uv,
        m_lambda_ss, m_lambda_ss_sum, m_uv_ss, and m_Elogprobw) will have a
        different ordering after execution of this method:  topics will
        be ordered so that more prominent topics are earlier.
        sstats are not re-ordered; it is assumed they will not be used
        after execution of this method.
        '''

        # rho will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rho = self.m_scale * pow(self.m_tau + self.m_t, -self.m_kappa)
        if rho < RHO_BOUND:
            rho = RHO_BOUND

        # Update lambda based on documents.
        self.m_lambda_ss *= (1 - rho)
        self.m_lambda_ss[:, word_list] += \
            rho * self.m_D * sstats.m_lambda_ss / sstats.m_batchsize
        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)

        self.m_t += 1

        self.m_uv_ss = (1.0 - rho) * self.m_uv_ss + rho * \
            sstats.m_uv_ss * self.m_D / sstats.m_batchsize

        self.m_Elogprobw = \
            sp.psi(self.m_eta + self.m_lambda_ss) - \
            sp.psi(self.m_W * self.m_eta + self.m_lambda_ss_sum[:, np.newaxis])

        if opt_o:
            self.optimal_ordering()

        # update top level sticks
        uv = np.zeros((2, self.m_K - 1))
        self.m_uv[0] = self.m_uv_ss[:self.m_K - 1] + 1.0
        var_phi_sum = np.flipud(self.m_uv_ss[1:])
        self.m_uv[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma

    def save_topics(self, filename):
        '''
        Write the topics (specified by variational means lambda + eta)
        to file.
        '''

        f = file(filename, "w")
        lambdas = self.m_lambda_ss + self.m_eta
        for lamb in lambdas:
            line = ' '.join([str(x) for x in lamb])
            f.write(line + '\n')
        f.close()

    def hdp_to_lda(self):
        '''
        Compute the LDA model corresponding to this HDP model.
        '''

        # alpha
        uv = self.m_uv[0] / (self.m_uv[0] + self.m_uv[1])
        alpha = np.zeros(self.m_K)
        left = 1.0
        for i in range(0, self.m_K - 1):
            alpha[i] = uv[i] * left
            left = left - alpha[i]
        alpha[self.m_K - 1] = left
        alpha = alpha * self.m_alpha
        #alpha = alpha * self.m_gamma

        # beta
        beta = (self.m_lambda_ss + self.m_eta) / (self.m_W * self.m_eta +
                                               self.m_lambda_ss_sum[:, np.newaxis])

        return (alpha, beta)

    def infer_only(self, docs, half_train_half_test=False, split_ratio=0.9, iterative_average=False):
        '''
        Infer topic assignments for the given documents, fixing the
        current topic prior.  Return a (likelihood, token_count) pair.
        '''

        uv = self.m_uv[0] / (self.m_uv[0] + self.m_uv[1])
        alpha = np.zeros(self.m_K)
        left = 1.0
        for i in range(0, self.m_K - 1):
            alpha[i] = uv[i] * left
            left = left - alpha[i]
        alpha[self.m_K - 1] = left
        #alpha = alpha * self.m_gamma
        score = 0.0
        count = 0.0
        for doc in docs:
            if half_train_half_test:
                (s, c, gamma) = lda_e_step_half(
                    doc, alpha, self.m_Elogprobw, split_ratio)
                score += s
                count += c
            else:
                score += lda_e_step(doc, alpha, np.exp(self.m_Elogprobw))
                count += doc.total
        return (score, count)


def lda_e_step_half(doc, alpha, Elogprobw, split_ratio):
    n_train = int(doc.length * split_ratio)
    n_test = doc.length - n_train

   # split the document
    words_train = doc.words[:n_train]
    counts_train = doc.counts[:n_train]
    words_test = doc.words[n_train:]
    counts_test = doc.counts[n_train:]

    gamma = np.ones(len(alpha))
    expElogtheta = np.exp(utils.dirichlet_expectation(gamma))

    expElogprobw = np.exp(Elogprobw)
    expElogprobw_train = expElogprobw[:, words_train]
    phinorm = np.dot(expElogtheta, expElogprobw_train) + 1e-100
    counts = np.array(counts_train)
    iter = 0
    max_iter = 100
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        gamma = alpha + expElogtheta * \
            np.dot(counts / phinorm, expElogprobw_train.T)
        Elogtheta = utils.dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, expElogprobw_train) + 1e-100
        meanchange = np.mean(abs(gamma - lastgamma))
        if (meanchange < MEANCHANGETHRESH):
            break
    gamma = gamma / np.sum(gamma)
    counts = np.array(counts_test)
    expElogprobw_test = expElogprobw[:, words_test]
    score = np.sum(counts * np.log(np.dot(gamma, expElogprobw_test) + 1e-100))

    return (score, np.sum(counts), gamma)


def lda_e_step_split(doc, alpha, beta, max_iter=100):
    half_len = int(doc.length / 2) + 1
    idx_train = [2 * i for i in range(half_len) if 2 * i < doc.length]
    idx_test = [2 * i + 1 for i in range(half_len) if 2 * i + 1 < doc.length]

   # split the document
    words_train = [doc.words[i] for i in idx_train]
    counts_train = [doc.counts[i] for i in idx_train]
    words_test = [doc.words[i] for i in idx_test]
    counts_test = [doc.counts[i] for i in idx_test]

    gamma = np.ones(len(alpha))
    expElogtheta = np.exp(utils.dirichlet_expectation(gamma))
    betad = beta[:, words_train]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(counts_train)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts / phinorm,  betad.T)
        Elogtheta = utils.dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, betad) + 1e-100
        meanchange = np.mean(abs(gamma - lastgamma))
        if (meanchange < MEANCHANGETHRESH):
            break

    gamma = gamma / np.sum(gamma)
    counts = np.array(counts_test)
    betad = beta[:, words_test]
    score = np.sum(counts * np.log(np.dot(gamma, betad) + 1e-100))

    return (score, np.sum(counts), gamma)


def lda_e_step(doc, alpha, beta, max_iter=100):
    gamma = np.ones(len(alpha))
    expElogtheta = np.exp(utils.dirichlet_expectation(gamma))
    betad = beta[:, doc.words]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(doc.counts)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts / phinorm,  betad.T)
        Elogtheta = utils.dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, betad) + 1e-100
        meanchange = np.mean(abs(gamma - lastgamma))
        if (meanchange < MEANCHANGETHRESH):
            break

    likelihood = np.sum(counts * np.log(phinorm))
    likelihood += np.sum((alpha - gamma) * Elogtheta)
    likelihood += np.sum(sp.gammaln(gamma) - sp.gammaln(alpha))
    likelihood += sp.gammaln(np.sum(alpha)) - sp.gammaln(np.sum(gamma))

    return (likelihood, gamma)
