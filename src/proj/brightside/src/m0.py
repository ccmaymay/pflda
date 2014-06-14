'''
Online variational inference for nHDP with several tricks to improve
performance.  Part of code is adapted from Matt's online LDA code.
'''

import logging
import numpy as np
import numpy.linalg as la
import numpy.random as nprand
import scipy.special as sp
import itertools as it
import utils
import random


# TODO assert var beta/dirichlet parameters no smaller than prior


MEANCHANGETHRESH = 0.00001
RANDOM_SEED = 999931111
MIN_ADDING_NOISE_POINT = 10
MIN_ADDING_NOISE_RATIO = 1
MU0 = 0.3


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class suff_stats:

    '''
    Struct for per-document sufficient statistics for one or more
    documents.

    Members
    -------
    m_batchsize:      int, number of documents this object represents
    m_tau_ss:  (K) float, each entry of which is the sum of
                      var_phi over all tokens in the document(s)
                      allocated the corresponding (global) topic
    m_lambda_ss:    (K x Wt) float, each entry of which is the sum of
                      var_phi * nu over all tokens of the corresponding
                      type in the document(s) that are allocated the
                      corresponding (global) topic.  Wt is the number
                      of word types spanned by the Dt document(s).
    '''

    def __init__(self, K, Wt, Dt):
        self.m_batchsize = Dt
        self.m_tau_ss = np.zeros(K)
        self.m_lambda_ss = np.zeros((K, Wt))

    def set_zero(self):
        self.m_tau_ss.fill(0.0)
        self.m_lambda_ss.fill(0.0)


class m0:

    r'''
    nHDP model using stick breaking.

    This class implements the following tricks not described in the
    online nHDP paper:
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
        \rho_t = \max{(a, b (\iota_0 + t)^{-\kappa})}
      \]
      where a is the lower bound (member m_rho_bound) and b is the
      scaling factor (member m_scale) (see below).

    Members
    -------
    m_trunc:              tuple, truncations (per level)
    m_depth:              int, max number of levels after truncation
    m_K:                  int, max number of topics after truncation
    m_W:                  int, size of vocabulary
    m_D:                  int, size of corpus (in documents)
    m_kappa:              float, learning rate
    m_iota:                float, slow down parameter
    m_delta:              float, stopping criterion of greedy
                          subtree selection algorithm (ELBO threshold)
    m_lambda_ss:             (K x W) float, variational parameters of
                          top-level topics (lambda in paper; top-level
                          topics are phi)
    m_lambda0:                float, parameter on top-level topic Dirichlet
    m_beta:              float, second-level DP concentration parameter
    m_alpha:              float, first-level DP concentration parameter
    m_gamma1:              float, switching probability Beta shape param
    m_gamma2:              float, switching probability Beta shape param
    m_tau:         (2 x K-1) float, variational parameters of
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
    m_tau_ss           (K) float, sufficient statistics for global
                          relative stick length variational parameters
                          (corresponds to suff_stats.m_tau_ss)
    m_lambda_ss_sum:         (K) float, sum of word proportions lambda
                          (per topic)
    '''

    def __init__(self, trunc, D, W, lambda0=0.01, beta=1., alpha=1., gamma1=1./3., gamma2=2./3., kappa=0.5, iota=1., delta=1e-3, scale=1., rho_bound=0., adding_noise=False):
        if trunc[0] != 1:
            raise ValueError('Top-level truncation must be one.')

        self.m_trunc = trunc
        self.m_trunc_idx_b = utils.tree_index_b(trunc)
        self.m_trunc_idx_m = utils.tree_index_m(trunc)

        self.m_K = int(np.sum(self.m_trunc_idx_b)) # total no. nodes
        self.m_W = W
        self.m_D = D
        self.m_depth = len(trunc)
        self.m_beta = beta
        self.m_alpha = alpha
        self.m_gamma1 = gamma1
        self.m_gamma2 = gamma2

        self.m_tau = np.zeros((2, self.m_K))
        self.m_tau[0] = 1.0
        #self.m_tau[1] = self.m_alpha
        # make a uniform at beginning
        # TODO why? and how is this uniform?
        self.m_tau[1] = range(self.m_K, 0, -1)
        for global_node in self.tree_iter():
            global_node_idx = self.tree_index(global_node)
            if global_node[-1] + 1 == self.m_trunc[len(global_node)-1]: # right-most child in truncated tree
                self.m_tau[0,global_node_idx] = 1.0
                self.m_tau[1,global_node_idx] = 0.0

        self.m_tau_ss = np.zeros(self.m_K)

        # Intuition: take 100 to be the expected document length (TODO)
        # so that there are 100D tokens in total.  Then divide that
        # count somewhat evenly (i.i.d. Gamma(1,1) distributed) between
        # each word type and topic.  *Then* subtract lambda0 so that the
        # posterior is composed of these pseudo-counts only (maximum
        # likelihood / no prior).  (why?!  TODO)
        self.m_lambda_ss = np.random.gamma(
            1.0, 1.0, (self.m_K, W)) * D * 100 / (self.m_K * W) - lambda0
        self.m_lambda0 = lambda0
        self.m_Elogprobw = utils.log_dirichlet_expectation(self.m_lambda0 + self.m_lambda_ss)

        self.m_iota = iota
        self.m_kappa = kappa
        self.m_delta = delta
        self.m_scale = scale
        self.m_rho_bound = rho_bound
        self.m_t = 0
        self.m_adding_noise = adding_noise
        self.m_num_docs_parsed = 0

        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)

    def initialize(self, c, xi, burn_in_samples=None, omicron=None):
        '''
        Initialize m_lambda_ss and m_Elogprobw (and m_lambda_ss_sum) using
        five E-step trials on each of the provided documents.  (Use
        burn_in_samples of the documents, if not None, else use all
        provided documents.)
        '''

        if burn_in_samples is None:
            num_samples = c.num_docs
        else:
            num_samples = min(c.num_docs, burn_in_samples)

        if omicron is None:
            omicron = num_samples

        ids = random.sample(range(c.num_docs), num_samples)
        docs = [c.docs[id] for id in ids]

        vocab_to_batch_word_map = dict()
        batch_to_vocab_word_map = []
        for doc in docs:
            for w in doc.words:
                if w not in vocab_to_batch_word_map:
                    vocab_to_batch_word_map[w] = len(vocab_to_batch_word_map)
                    batch_to_vocab_word_map.append(w)
        Wt = len(batch_to_vocab_word_map)  # number of unique words in these documents

        kmeans_data = [
            (
                [vocab_to_batch_word_map[w] for w in doc.words],
                np.array(doc.counts) / float(sum(doc.counts))
            )
            for doc in docs
        ]

        logging.debug('Initialization means:')
        cluster_assignments = np.zeros(num_samples, dtype=np.uint)
        cluster_means = np.zeros((self.m_K, Wt)) # TODO big!
        for node in self.tree_iter():
            if len(node) < len(self.m_trunc): # not leaf
                idx = self.tree_index(node)
                num_children = self.m_trunc[len(node)]
                c_ids = np.zeros(num_children, dtype=np.uint)
                for j in xrange(num_children):
                    c_node = node + (j,)
                    c_ids[j] = self.tree_index(c_node)
                node_doc_ids = np.where(cluster_assignments == idx)[0]
                if len(node_doc_ids) > 0:
                    node_kmeans_data = [kmeans_data[i] for i in node_doc_ids]
                    (node_cluster_assignments, node_cluster_means) = utils.kmeans_sparse(
                        node_kmeans_data, Wt, num_children, norm=1)
                    cluster_assignments[node_doc_ids] = c_ids[node_cluster_assignments]
                    cluster_means[c_ids,:] = node_cluster_means
                    logging.debug('Node %s:' % str(node))
                    for i in xrange(num_children):
                        w_order = np.argsort(cluster_means[c_ids[i],:])
                        logging.debug('\t%s' % ' '.join(str(batch_to_vocab_word_map[w_order[j]]) for j in xrange(Wt-1, max(-1,Wt-11), -1)))
                    for i in xrange(len(node_doc_ids)):
                        x = kmeans_data[i]
                        cluster = cluster_assignments[node_doc_ids[i]]
                        # note, here we are only computing the
                        # difference between the non-zero components
                        # of x[1] and the corresponding components of
                        # cluster_means... but this is okay because we
                        # assume x[1] > 0, so cluster_means > 0, so when
                        # we threshold (below), the differences of all
                        # components where x[1] is zero will be zeroed.
                        new_features = x[1] - cluster_means[cluster,x[0]]
                        new_features[new_features < 0] = 0
                        nf_sum = np.sum(new_features)
                        if nf_sum > 0:
                            new_features /= nf_sum
                        kmeans_data[i] = (x[0], new_features)
                else:
                    logging.debug('Node %d: no docs' % idx)
                    cluster_means[c_ids,:] = cluster_means[idx,:]

        self.m_lambda_ss = omicron * (1 - xi) * nprand.dirichlet(100 * np.ones(self.m_W) / float(self.m_W), self.m_K)
        self.m_lambda_ss[:, batch_to_vocab_word_map] += omicron * xi * cluster_means

        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)
        self.m_Elogprobw = utils.log_dirichlet_expectation(self.m_lambda0 + self.m_lambda_ss)

    def process_documents(self, docs, var_converge, unseen_ids=None, update=True, predict_docs=None):
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
        if unseen_ids is None:
            unseen_ids = []

        if predict_docs is None:
            predict_docs = [None] * len(docs)

        # Find the unique words in this mini-batch of documents...
        self.m_num_docs_parsed += len(docs)
        adding_noise = False
        adding_noise_point = MIN_ADDING_NOISE_POINT

        if self.m_adding_noise:
            if float(adding_noise_point) / len(docs) < MIN_ADDING_NOISE_RATIO:
                adding_noise_point = MIN_ADDING_NOISE_RATIO * len(docs)

            if self.m_num_docs_parsed % adding_noise_point == 0:
                adding_noise = True

        # mapping from word types in this mini-batch to unique
        # consecutive integers
        vocab_to_batch_word_map = dict()
        # list of unique word types, in order of first appearance
        batch_to_vocab_word_map = []
        if adding_noise:
            batch_to_vocab_word_map = range(self.m_W)
            for w in batch_to_vocab_word_map:
                vocab_to_batch_word_map[w] = w
        else:
            for doc in docs:
                for w in doc.words:
                    if w not in vocab_to_batch_word_map:
                        vocab_to_batch_word_map[w] = len(vocab_to_batch_word_map)
                        batch_to_vocab_word_map.append(w)

        # number of unique word types in this mini-batch
        num_tokens = sum([sum(doc.counts) for doc in docs])
        Wt = len(batch_to_vocab_word_map)

        logging.info('Processing %d docs spanning %d tokens, %d types'
            % (len(docs), num_tokens, Wt))

        ss = suff_stats(self.m_K, Wt, len(docs))

        # First row of ElogV is E[log(V)], second row is E[log(1 - V)]
        psi_sum_tau = sp.psi(np.sum(self.m_tau, 0))
        ElogV = sp.psi(self.m_tau) - psi_sum_tau

        # run variational inference on some new docs
        score = 0.0
        count = 0
        unseen_score = 0.0
        unseen_count = 0
        for i, (doc, predict_doc) in enumerate(it.izip(docs, predict_docs)):
            doc_score = self.doc_e_step(doc, ss, ElogV, vocab_to_batch_word_map,
                                        batch_to_vocab_word_map, var_converge,
                                        predict_doc=predict_doc)
            score += doc_score
            if predict_doc is None:
                count += doc.total
            else:
                count += predict_doc.total
            if i in unseen_ids:
                unseen_score += doc_score
                if predict_doc is None:
                    unseen_count += doc.total
                else:
                    unseen_count += predict_doc.total

        if adding_noise:
            # add noise to the ss
            logging.debug("Adding noise")

            # old noise
            noise = np.random.gamma(1.0, 1.0, ss.m_lambda_ss.shape)
            noise_sum = np.sum(noise, axis=1)
            ratio = np.sum(ss.m_lambda_ss, axis=1) / noise_sum
            noise = noise * ratio[:, np.newaxis]

            # new noise
            #lambda_sum_tmp = self.m_W * self.m_lambda0 + self.m_lambda_ss_sum
            #scaled_beta = 5*self.m_W * (self.m_lambda_ss + self.m_lambda0) / (lambda_sum_tmp[:, np.newaxis])
            #noise = np.random.gamma(shape=scaled_beta, scale=1.0)
            #noise_ratio = lambda_sum_tmp / noise_sum
            #noise = (noise * noise_ratio[:, np.newaxis] - self.m_lambda0) * len(docs)/self.m_D

            mu = MU0 * 1000.0 / (self.m_t + 1000)

            ss.m_lambda_ss = ss.m_lambda_ss * (1.0 - mu) + noise * mu

        if update:
            self.update_ss_stochastic(ss, batch_to_vocab_word_map)
            self.update_lambda()
            self.update_tau()
            self.m_t += 1

        return (score, count, unseen_score, unseen_count)

    def update_lambda(self):
        self.m_Elogprobw = (
            sp.psi(self.m_lambda0 + self.m_lambda_ss)
            - sp.psi(self.m_W*self.m_lambda0 + self.m_lambda_ss_sum[:, np.newaxis])
        )

    def update_nu(self, subtree, ab, uv, Elogprobw_doc, doc, nu, log_nu):
        Elogpi = np.zeros(self.m_K)
        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)

                # contributions from switching probabilities
                if idx == p_idx:
                    Elogpi[idx] += (
                        sp.psi(ab[0, p_idx])
                        - sp.psi(np.sum(ab[:, p_idx]))
                    )
                else:
                    Elogpi[idx] += (
                        sp.psi(ab[1, p_idx])
                        - sp.psi(np.sum(ab[:, p_idx]))
                    )

                # contributions from stick proportions
                Elogpi[idx] += (
                    sp.psi(uv[0, p_idx])
                    - sp.psi(np.sum(uv[:, p_idx]))
                )
                for s in self.node_left_siblings(p):
                    s_idx = self.tree_index(s)
                    Elogpi[idx] += (
                        sp.psi(uv[1, s_idx])
                        - sp.psi(np.sum(uv[:, s_idx]))
                    )

        # TODO oHDP: only add Elogpi if iter < 3
        log_nu[:,:] = np.repeat(Elogprobw_doc, doc.counts, axis=1).T + Elogpi # N x K
        log_nu[:,[self.tree_index(node) for node in self.tree_iter() if node not in subtree]] = -np.inf
        (log_nu[:,:], log_norm) = utils.log_normalize(log_nu)
        nu[:,:] = np.exp(log_nu)

    def update_uv(self, subtree, nu_sums, uv):
        '''
        Update uv in-place.
        Fix q(V^{(d)}_i = 1) = 1 for all i that are the right-most
        children of their parent.
        '''
        uv[0] = 1.0
        uv[1] = self.m_beta
        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            if node[:-1] + (node[-1] + 1,) not in subtree: # last child of this node in subtree
                uv[:,idx] = [1.0, 0.0]
            for p in it.chain((node,), self.node_ancestors(node)):
                if len(p) > 1: # not root
                    p_idx = self.tree_index(p)
                    if p[:-1] + (p[-1] + 1,) in subtree:
                        uv[0,p_idx] += nu_sums[idx]

                    # left siblings of this ancestor
                    for s in self.node_left_siblings(p):
                        s_idx = self.tree_index(s)
                        uv[1,s_idx] += nu_sums[idx]

    def update_ab(self, subtree, nu_sums, ab):
        '''
        Update ab in-place.
        Fix q(U_{di} = 1) = 1 for all i that are leaves of the truncated
        subtree.
        '''
        ab[0] = self.m_gamma1 + nu_sums
        ab[1] = self.m_gamma2
        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            if node + (0,) not in subtree: # leaf in subtree
                ab[:,idx] = [1.0, 0.0]
            for p in self.node_ancestors(node):
                p_idx = self.tree_index(p)
                ab[1,p_idx] += nu_sums[idx]

    def update_tau(self):
        '''
        Update self.m_tau in-place.
        Fix q(V_j = 1) = 1 for all j that are the right-most children
        of their parent.
        '''
        self.m_tau[0] = self.m_tau_ss + 1.0
        self.m_tau[1] = self.m_alpha
        for global_node in self.tree_iter():
            global_node_idx = self.tree_index(global_node)
            if global_node[-1] + 1 == self.m_trunc[len(global_node)-1]: # right-most child in truncated tree
                self.m_tau[0,global_node_idx] = 1.0
                self.m_tau[1,global_node_idx] = 0.0
            for global_s in self.node_left_siblings(global_node):
                global_s_idx = self.tree_index(global_s)
                self.m_tau[1,global_s_idx] += self.m_tau_ss[global_node_idx]

    def z_likelihood(self, subtree, ElogV):
        '''
        Return E[log p(z | V)] + H(q(z)).  (Note H(q(z)) = 0.)
        Assume ElogV columns corresponding to the right-most child of
        each node (*globally*) are [0, inf].
        '''
        self.check_ElogV_edge_cases(ElogV)

        likelihood = 0.0
        for node in self.tree_iter(subtree):
            global_node = subtree[node]
            global_idx = self.tree_index(global_node)
            likelihood += ElogV[0,global_idx]
            for global_s in self.node_left_siblings(global_node):
                global_s_idx = self.tree_index(global_s)
                likelihood += ElogV[1,global_s_idx]
        return likelihood

    def c_likelihood(self, subtree, ab, uv, nu, log_nu, ids):
        '''
        Return E[log p(c | U, V)] + H(q(c)).
        Assume ab[:,i] = [1, 0] for i leaf and uv[:,i] = [1, 0] for
        i right-most child of its parent node.
        '''
        self.check_ab_edge_cases(ab, subtree, ids)
        self.check_uv_edge_cases(uv, subtree, ids)
        self.check_nu_edge_cases(nu)
        self.check_log_nu_edge_cases(log_nu)
        self.check_subtree_ids(subtree, ids)

        log_prob_c = np.zeros(self.m_K)
        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            log_prob_c[idx] += (
                sp.psi(ab[0,idx])
                - sp.psi(np.sum(ab[:,idx]))
            )
            if len(node) > 1: # not root
                for p in it.chain((node,), self.node_ancestors(node)):
                    p_idx = self.tree_index(p)
                    if len(p) < len(node): # equivalent to: p != node
                        log_prob_c[idx] += (
                            sp.psi(ab[1,p_idx])
                            - sp.psi(np.sum(ab[:,p_idx]))
                        )
                    log_prob_c[idx] += (
                        sp.psi(uv[0,p_idx])
                        - sp.psi(np.sum(uv[:,p_idx]))
                    )
                    for s in self.node_left_siblings(p):
                        s_idx = self.tree_index(s)
                        log_prob_c[idx] += (
                            sp.psi(uv[1,s_idx])
                            - sp.psi(np.sum(uv[:,s_idx]))
                        )

        assert (log_prob_c <= 0).all()
        return np.sum(nu[:,ids] * (log_prob_c[ids][np.newaxis,:] - log_nu[:,ids]))

    def w_likelihood(self, doc, nu, Elogprobw_doc, ids):
        '''
        Return E[log p(W | theta, c, z)].
        Assume rows of nu sum to one.
        '''
        self.check_nu_edge_cases(nu)

        return np.sum(nu[:,ids].T * np.repeat(Elogprobw_doc[ids,:], doc.counts, axis=1))

    def check_subtree_ids(self, subtree, ids):
        ids_in_subtree = set(ids)
        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            assert idx in ids_in_subtree, 'id %d in subtree but not in id list' % idx
            ids_in_subtree.remove(idx)
        assert not ids_in_subtree, 'ids in id list but not in subtree: %s' % str(ids_in_subtree)

    def check_ab_edge_cases(self, ab, subtree, ids):
        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            if idx in ids and node + (0,) not in subtree: # leaf in subtree
                assert ab[0, idx] == 1. and ab[1, idx] == 0., 'leaf %s has ab = %s (require [1, 0])' % (str(node), str(ab[:, idx]))

    def check_uv_edge_cases(self, uv, subtree, ids):
        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            s = node[:-1] + (node[-1] + 1,) # right child
            if idx in ids and s not in subtree: # node is last child of its parent in subtree
                assert uv[0, idx] == 1. and uv[1, idx] == 0., 'right-most child %s has uv = %s (require [1, 0])' % (str(node), str(uv[:, idx]))

    def check_ElogV_edge_cases(self, ElogV):
        for node in self.tree_iter():
            idx = self.tree_index(node)
            if node[-1] + 1 == self.m_trunc[len(node)-1]: # node is last child of its parent in global tree
                assert ElogV[0, idx] == 0. and ElogV[1, idx] == np.inf, 'right-most child %s has ElogV = %s (require [0, inf])' % (str(node), str(ElogV[:, idx]))

    def check_nu_edge_cases(self, nu):
        assert la.norm(np.sum(nu,1) - 1, np.inf) < 1e-9, 'not all rows of nu sum to one: %s' % str(np.sum(nu, 1))

    def check_log_nu_edge_cases(self, log_nu):
        assert la.norm(np.sum(np.exp(log_nu),1) - 1, np.inf) < 1e-9, 'not all rows of exp(log_nu) sum to one: %s' % str(np.sum(np.exp(log_nu),1))

    def doc_e_step(self, doc, ss, ElogV, vocab_to_batch_word_map,
                   batch_to_vocab_word_map, var_converge, max_iter=100,
                   predict_doc=None):
        '''
        Perform document-level coordinate ascent updates of variational
        parameters.  Don't incorporate variational expectations of log
        stick lengths until the third iteration of the E-step (and
        later).  Update global sufficient statistics by incrementing
        members of ss accordingly.  Return likelihood for this E-step.

        Variables
        ---------
        uv:               (2 x T-1) float, variational parameters of
                         second level relative stick lengths (a and b in
                         paper, respectively by row; rel stick lengths
                         are pi')
        nu:             (N x T) float, variational parameters for topic
                         index of token (index into active topics for
                         this document) (nu in paper; indices are z)
        Elogprobw_doc:    (K x N) float, expected log word likelihoods
                         (E log p(w | phi) in paper)
        ElogV:  (K) float, expected log first-level stick
                         lengths (E log beta in paper)
        Elogpi:  (T) float, expected log second-level stick
                         lengths (E log pi in paper)

        (denote by N the number of *types* in the document)
        '''

        num_tokens = sum(doc.counts)

        logging.debug('Performing E-step on doc spanning %d tokens, %d types'
            % (num_tokens, len(doc.words)))

        # each position of this list represents a token in our document;
        # the value in that position is the word type id specific to
        # this mini-batch of documents (a unique integer between zero
        # and the number of types in this mini-batch)
        batch_ids = [vocab_to_batch_word_map[w] for w in doc.words]
        token_batch_ids = np.repeat(batch_ids, doc.counts)

        (subtree, l2g_idx, g2l_idx) = self.select_subtree(doc, ElogV, num_tokens)
        ids = [self.tree_index(node) for node in self.tree_iter(subtree)]

        Elogprobw_doc = self.m_Elogprobw[l2g_idx, :][:, doc.words]

        logging.debug('Initializing document variational parameters')

        # uniform
        nu = np.zeros((num_tokens, self.m_K))
        nu[:, ids] = 1.0 / float(len(subtree))
        log_nu = np.log(nu)
        nu_sums = np.sum(nu, 0)

        # q(V_{i,j}^{(d)} = 1) = 1 for j+1 = trunc[\ell] (\ell is depth)
        uv = np.zeros((2, self.m_K))
        uv[0] = 1.0
        uv[1] = self.m_beta
        uv_ids = []
        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            s = node[:-1] + (node[-1] + 1,) # right child
            if s not in subtree: # node is last child of its parent in subtree
                uv[0,idx] = 1.0
                uv[1,idx] = 0.0
            else:
                uv_ids.append(idx)

        ab = np.zeros((2, self.m_K))
        ab[0] = self.m_gamma1
        ab[1] = self.m_gamma2
        ab_ids = []
        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            if node + (0,) not in subtree: # leaf in subtree
                ab[0,idx] = 1.0
                ab[1,idx] = 0.0
            else:
                ab_ids.append(idx)

        converge = None
        likelihood = None
        old_likelihood = None

        iteration = 0
        # not yet support second level optimization yet, to be done in the
        # future
        while iteration < max_iter and (converge is None or converge < 0.0 or converge > var_converge):
            logging.debug('Updating document variational parameters (iteration: %d)' % iteration)
            # update variational parameters

            self.update_nu(subtree, ab, uv, Elogprobw_doc, doc, nu, log_nu)
            nu_sums = np.sum(nu, 0)
            self.update_uv(subtree, nu_sums, uv)
            self.update_ab(subtree, nu_sums, ab)

            # compute likelihood

            likelihood = 0.0

            # E[log p(U | gamma_1, gamma_2)] + H(q(U))
            u_ll = utils.log_sticks_likelihood(ab, self.m_gamma1, self.m_gamma2, ab_ids)
            likelihood += u_ll
            logging.debug('Log-likelihood after U components: %f (+ %f)' % (likelihood, u_ll))

            # E[log p(V | beta)] + H(q(V))
            v_ll = utils.log_sticks_likelihood(uv, 1.0, self.m_beta, uv_ids)
            likelihood += v_ll
            logging.debug('Log-likelihood after V components: %f (+ %f)' % (likelihood, v_ll))

            # E[log p(z | V)] + H(q(z))  (note H(q(z)) = 0)
            z_ll = self.z_likelihood(subtree, ElogV)
            likelihood += z_ll
            logging.debug('Log-likelihood after z components: %f (+ %f)' % (likelihood, z_ll))

            # E[log p(c | U, V)] + H(q(c))
            # TODO is it a bug that the equivalent computation in
            # oHDP does not account for types appearing more than
            # once?  (Uses . rather than ._all .)
            c_ll = self.c_likelihood(subtree, ab, uv, nu, log_nu, ids)
            likelihood += c_ll
            logging.debug('Log-likelihood after c components: %f (+ %f)' % (likelihood, c_ll))

            # E[log p(W | theta, c, z)]
            w_ll = self.w_likelihood(doc, nu, Elogprobw_doc, ids)
            likelihood += w_ll
            logging.debug('Log-likelihood after W component: %f (+ %f)' % (likelihood, w_ll))

            logging.debug('Log-likelihood: %f' % likelihood)

            if old_likelihood is not None:
                converge = (likelihood - old_likelihood) / abs(old_likelihood)
                if converge < 0:
                    logging.warning('Log-likelihood is decreasing')
            old_likelihood = likelihood

            iteration += 1

        # update the suff_stat ss
        global_ids = l2g_idx[ids]
        ss.m_tau_ss[global_ids] += 1
        for n in xrange(num_tokens):
            ss.m_lambda_ss[global_ids, token_batch_ids[n]] += nu[n, ids]

        if predict_doc is not None:
            logEpi = np.zeros(self.m_K)

            # TODO precompute ratios, here and elsewhere?
            for node in self.tree_iter(subtree):
                idx = self.tree_index(node)
                for p in it.chain((node,), self.node_ancestors(node)):
                    p_idx = self.tree_index(p)

                    # contributions from switching probabilities
                    if idx == p_idx:
                        logEpi[idx] += np.log(ab[0, p_idx]) - np.log(np.sum(ab[:, p_idx]))
                    else:
                        logEpi[idx] += np.log(ab[1, p_idx]) - np.log(np.sum(ab[:, p_idx]))

                    # contributions from stick proportions
                    logEpi[idx] += np.log(uv[0, p_idx]) - np.log(np.sum(uv[:, p_idx]))
                    for s in self.node_left_siblings(p):
                        s_idx = self.tree_index(s)
                        logEpi[idx] += np.log(uv[1, s_idx]) - np.log(np.sum(uv[:, s_idx]))

            logEtheta = (
                np.log(self.m_lambda0 + self.m_lambda_ss)
                - np.log(self.m_W*self.m_lambda0 + self.m_lambda_ss_sum[:,np.newaxis])
            )

            likelihood = np.sum(np.log(np.sum(np.exp(logEpi[ids][:,np.newaxis] + logEtheta[l2g_idx[ids],:][:,predict_doc.words]), 0)) * predict_doc.counts)

        return likelihood

    def node_ancestors(self, node):
        r'''
        Return generator over this node's ancestors, starting from the
        parent and ascending upward.
        '''
        return utils.node_ancestors(node)

    def node_left_siblings(self, node):
        r'''
        Return generator over this node's elder siblings (children of the
        same parent, left of this node in the planar embedding).  Generator
        starts at the sibling to the immediate left and proceeds leftward.
        '''
        return utils.node_left_siblings(node)

    def tree_iter(self, subtree=None):
        '''
        Return generator over all nodes in tree, ordered first by level,
        and then left-to-right within each level.  If subtree is not
        None, the generator is filtered to only return nodes in the
        subtree (as determined by subtree.__contains__).
        '''
        if subtree is None:
            return utils.tree_iter(self.m_trunc)
        else:
            return (n for n in utils.tree_iter(self.m_trunc) if n in subtree)

    def tree_index(self, x):
        '''
        Return index into flat array representing the tree corresponding
        to level-wise offsets given in x.  See utils.tree_index for details.
        '''
        return utils.tree_index(x, self.m_trunc_idx_m, self.m_trunc_idx_b)

    def subtree_node_candidates(self, subtree):
        '''
        TODO
        '''

        return utils.subtree_node_candidates(self.m_trunc, subtree)

    def select_subtree(self, doc, ElogV, num_tokens):
        # TODO abstract stuff below, like subtree candidate
        # modifications... prone to bugs
        logging.debug('Greedily selecting subtree for ' + str(doc.identifier))

        # map from local nodes in subtree to global nodes
        subtree = dict()
        subtree[(0,)] = (0,) # root
        # map from local (subtree) node indices to global indices
        l2g_idx = np.zeros(self.m_K, dtype=np.uint)
        # map from global node indices to local (subtree) node indices;
        # unmapped global nodes are left at zero (note that there is no
        # ambiguity here, as the local root only maps to the global
        # root and vice-versa)
        g2l_idx = np.zeros(self.m_K, dtype=np.uint)

        prior_uv = np.zeros((2, self.m_K))
        prior_uv[0] = 1.0

        prior_ab = np.zeros((2, self.m_K))
        prior_ab[0] = 1.0

        old_likelihood = 0.0

        ids = [self.tree_index(nod) for nod in self.tree_iter(subtree)]
        logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))

        Elogprobw_doc = self.m_Elogprobw[l2g_idx, :][:, doc.words]
        nu = np.zeros((num_tokens, self.m_K))
        log_nu = np.log(nu)
        self.update_nu(
            subtree, prior_ab, prior_uv, Elogprobw_doc, doc, nu, log_nu)

        # E[log p(z | V)] + H(q(z))  (note H(q(z)) = 0)
        z_ll = self.z_likelihood(subtree, ElogV)
        old_likelihood += z_ll
        logging.debug('Log-likelihood after z components: %f (+ %f)'
            % (old_likelihood, z_ll))

        # E[log p(c | U, V)] + H(q(c))
        # TODO is it a bug that the equivalent computation in
        # oHDP does not account for types appearing more than
        # once?  (Uses . rather than ._all .)
        c_ll = self.c_likelihood(subtree, prior_ab, prior_uv, nu, log_nu, ids)
        old_likelihood += c_ll
        logging.debug('Log-likelihood after c components: %f (+ %f)'
            % (old_likelihood, c_ll))

        # E[log p(W | theta, c, z)]
        w_ll = self.w_likelihood(doc, nu, self.m_Elogprobw[l2g_idx, :][:, doc.words], ids)
        old_likelihood += w_ll
        logging.debug('Log-likelihood after W component: %f (+ %f)'
            % (old_likelihood, w_ll))

        candidate_nu = np.zeros((num_tokens, self.m_K))
        candidate_log_nu = np.log(nu)

        while True:
            best_node = None
            best_global_node = None
            best_likelihood = None

            for (node, global_node) in self.subtree_node_candidates(subtree):
                logging.debug('Candidate: global node %s for local node %s'
                    % (str(global_node), str(node)))
                idx = self.tree_index(node)
                global_idx = self.tree_index(global_node)

                subtree[node] = global_node
                l2g_idx[idx] = global_idx
                p = node[:-1]
                if p in subtree and node[-1] == 0:
                    p_idx = self.tree_index(p)
                    prior_ab[:,p_idx] = [self.m_gamma1, self.m_gamma2]
                left_s = p + (node[-1] - 1,)
                if left_s in subtree:
                    left_s_idx = self.tree_index(left_s)
                    prior_uv[:,left_s_idx] = [1.0, self.m_beta]

                ids = [self.tree_index(nod) for nod in self.tree_iter(subtree)]
                logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))

                Elogprobw_doc = self.m_Elogprobw[l2g_idx, :][:, doc.words]
                self.update_nu(subtree, prior_ab, prior_uv, Elogprobw_doc, doc,
                    candidate_nu, candidate_log_nu)

                candidate_likelihood = 0.0

                # E[log p(z | V)] + H(q(z))  (note H(q(z)) = 0)
                z_ll = self.z_likelihood(subtree, ElogV)
                candidate_likelihood += z_ll
                logging.debug('Log-likelihood after z components: %f (+ %f)'
                    % (candidate_likelihood, z_ll))

                # E[log p(c | U, V)] + H(q(c))
                # TODO is it a bug that the equivalent computation in
                # oHDP does not account for types appearing more than
                # once?  (Uses . rather than ._all .)
                c_ll = self.c_likelihood(subtree, prior_ab, prior_uv, candidate_nu, candidate_log_nu, ids)
                candidate_likelihood += c_ll
                logging.debug('Log-likelihood after c components: %f (+ %f)'
                    % (candidate_likelihood, c_ll))

                # E[log p(W | theta, c, z)]
                w_ll = self.w_likelihood(doc, candidate_nu, self.m_Elogprobw[l2g_idx, :][:, doc.words], ids)
                candidate_likelihood += w_ll
                logging.debug('Log-likelihood after W component: %f (+ %f)'
                    % (candidate_likelihood, w_ll))

                if best_likelihood is None or candidate_likelihood > best_likelihood:
                    best_node = node
                    best_global_node = global_node
                    best_likelihood = candidate_likelihood

                del subtree[node]
                l2g_idx[idx] = 0
                if p in subtree and node[-1] == 0:
                    p_idx = self.tree_index(p)
                    prior_ab[:,p_idx] = [1.0, 0.0]
                if left_s in subtree:
                    left_s_idx = self.tree_index(left_s)
                    prior_uv[:,left_s_idx] = [1.0, 0.0]

            if best_likelihood is None: # no candidates
                break

            converge = (best_likelihood - old_likelihood) / abs(old_likelihood)
            if converge < self.m_delta:
                break

            logging.debug('Selecting global node %s for local node %s'
                % (str(best_global_node), str(best_node)))
            logging.debug('Log-likelihood: %f' % best_likelihood)

            subtree[best_node] = best_global_node
            idx = self.tree_index(best_node)
            global_idx = self.tree_index(best_global_node)
            l2g_idx[idx] = global_idx
            g2l_idx[global_idx] = idx
            p = best_node[:-1]
            if p in subtree and best_node[-1] == 0:
                p_idx = self.tree_index(p)
                prior_ab[:,p_idx] = [self.m_gamma1, self.m_gamma2]
            left_s = p + (best_node[-1] - 1,)
            if left_s in subtree:
                left_s_idx = self.tree_index(left_s)
                prior_uv[:,left_s_idx] = [1.0, self.m_beta]
            likelihood = best_likelihood

            logging.debug('Subtree ids: %s'
                % ' '.join(str(self.tree_index(nod))
                           for nod in self.tree_iter(subtree)))

            old_likelihood = likelihood

        logging.debug('Log-likelihood: %f' % old_likelihood)
        logging.info('Subtree global node ids for %s: %s'
            % (str(doc.identifier),
               ' '.join(str(l2g_idx[self.tree_index(nod)])
                        for nod in self.tree_iter(subtree))))

        return (subtree, l2g_idx, g2l_idx)

    def update_ss_stochastic(self, ss, batch_to_vocab_word_map):
        '''
        Perform stochastic update of sufficient statistics for global
        variational parameters, incorporating batch-wise sufficient
        statistics in ss.
        '''

        # rho will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        # Add one because self.m_t is zero-based.
        rho = self.m_scale * pow(self.m_iota + self.m_t + 1, -self.m_kappa)
        if rho < self.m_rho_bound:
            rho = self.m_rho_bound

        # Update lambda based on documents.
        self.m_lambda_ss *= (1 - rho)
        self.m_lambda_ss[:, batch_to_vocab_word_map] += (
            rho * ss.m_lambda_ss * self.m_D / ss.m_batchsize
        )
        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)

        self.m_tau_ss = (
            (1.0 - rho) * self.m_tau_ss
            + rho * ss.m_tau_ss * self.m_D / ss.m_batchsize
        )

    def save_topics(self, filename):
        '''
        Write the topics (specified by variational means lambda + lambda0)
        to file.
        '''

        with open(filename, 'w') as f:
            lambdas = self.m_lambda_ss + self.m_lambda0
            for lamb in lambdas:
                line = ' '.join([str(x) for x in lamb])
                f.write(line + '\n')
