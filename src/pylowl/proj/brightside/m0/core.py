import logging
import numpy as np
import numpy.linalg as la
import numpy.random as nprand
import scipy.special as sp
import itertools as it
import pylowl.proj.brightside.utils as utils
import random
import cPickle


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class suff_stats(object):
    def __init__(self, K, Wt, Dt):
        self.m_batchsize = Dt
        self.m_tau_ss = np.zeros(K)
        self.m_lambda_ss = np.zeros((K, Wt))


class model(object):
    def __init__(self,
                 trunc,
                 D,
                 W,
                 lambda0=0.01,
                 beta=1.,
                 alpha=1.,
                 gamma1=1./3.,
                 gamma2=2./3.,
                 kappa=0.5,
                 iota=1.,
                 delta=1e-3,
                 scale=1.,
                 rho_bound=0.,
                 subtree_output_files=None):
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
        self.m_tau[1] = alpha
        # make a uniform at beginning
        # TODO why? and how is this uniform?
        # self.m_tau[1] = range(self.m_K, 0, -1)
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
        # score / no prior).  (why?!  TODO)
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
        self.m_num_docs_processed = 0

        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)

        if subtree_output_files is None:
            self.subtree_output_files = dict()
        else:
            self.subtree_output_files = subtree_output_files

    def initialize(self, docs, init_noise_weight, eff_init_samples=None):
        docs = list(docs)
        num_samples = len(docs)

        if eff_init_samples is None:
            eff_init_samples = num_samples

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
                node_doc_indices = np.where(cluster_assignments == idx)[0]
                if len(node_doc_indices) > 0:
                    node_kmeans_data = [kmeans_data[i] for i in node_doc_indices]
                    (node_cluster_assignments, node_cluster_means) = utils.kmeans_sparse(
                        node_kmeans_data, Wt, num_children, norm=1)

                    # sort by word counts (decreasing)
                    node_cluster_word_counts = np.zeros(num_children, dtype=np.uint)
                    for i in xrange(num_children):
                        node_cluster_docs = np.where(node_cluster_assignments == i)[0]
                        for j in node_cluster_docs:
                            node_cluster_word_counts[i] += docs[node_doc_indices[j]].total
                    reverse_cluster_order = np.argsort(node_cluster_word_counts)
                    cluster_order = reverse_cluster_order[::-1]
                    cluster_order_rindex = np.zeros(num_children, dtype=np.uint)
                    for i in xrange(num_children):
                        cluster_order_rindex[cluster_order[i]] = i
                    node_cluster_means[:] = node_cluster_means[cluster_order]
                    node_cluster_assignments[:] = cluster_order_rindex[node_cluster_assignments]

                    cluster_assignments[node_doc_indices] = c_ids[node_cluster_assignments]
                    cluster_means[c_ids,:] = node_cluster_means
                    logging.debug('Node %s:' % str(node))
                    for i in xrange(num_children):
                        w_order = np.argsort(cluster_means[c_ids[i],:])
                        logging.debug('\t%s' % ' '.join(str(batch_to_vocab_word_map[w_order[j]]) for j in xrange(Wt-1, max(-1,Wt-11), -1)))
                    for i in xrange(len(node_doc_indices)):
                        x = kmeans_data[i]
                        cluster = cluster_assignments[node_doc_indices[i]]
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

        self.m_lambda_ss = eff_init_samples * init_noise_weight * nprand.dirichlet(100 * np.ones(self.m_W) / float(self.m_W), self.m_K)
        self.m_lambda_ss[:, batch_to_vocab_word_map] += eff_init_samples * (1 - init_noise_weight) * cluster_means

        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)
        self.m_Elogprobw = utils.log_dirichlet_expectation(self.m_lambda0 + self.m_lambda_ss)

    def process_documents(self, docs, var_converge, update=True,
                          predict_docs=None, save_model=False):
        docs = list(docs)
        doc_count = len(docs)

        if predict_docs is None:
            predict_docs = [None] * doc_count

        # Find the unique words in this mini-batch of documents...
        self.m_num_docs_processed += doc_count

        # mapping from word types in this mini-batch to unique
        # consecutive integers
        vocab_to_batch_word_map = dict()
        # list of unique word types, in order of first appearance
        batch_to_vocab_word_map = []
        for doc in docs:
            for w in doc.words:
                if w not in vocab_to_batch_word_map:
                    vocab_to_batch_word_map[w] = len(vocab_to_batch_word_map)
                    batch_to_vocab_word_map.append(w)

        # number of unique word types in this mini-batch
        num_tokens = sum([sum(doc.counts) for doc in docs])
        Wt = len(batch_to_vocab_word_map)

        logging.info('Processing %d docs spanning %d tokens, %d types'
            % (doc_count, num_tokens, Wt))

        ss = suff_stats(self.m_K, Wt, doc_count)

        # First row of ElogV is E[log(V)], second row is E[log(1 - V)]
        ids = [self.tree_index(node) for node in self.tree_iter()]
        ElogV = utils.log_beta_expectation(self.m_tau)

        # run variational inference on some new docs
        score = 0.0
        count = 0
        for (doc, predict_doc) in it.izip(docs, predict_docs):
            doc_score = self.doc_e_step(doc, ss, ElogV, vocab_to_batch_word_map,
                batch_to_vocab_word_map, var_converge, predict_doc=predict_doc,
                save_model=save_model)

            score += doc_score
            if predict_doc is None:
                count += doc.total
            else:
                count += predict_doc.total

        if update:
            self.update_ss_stochastic(ss, batch_to_vocab_word_map)
            self.update_lambda()
            self.update_tau()
            self.m_t += 1

        return (score, count, doc_count)

    def update_lambda(self):
        self.m_Elogprobw = (
            sp.psi(self.m_lambda0 + self.m_lambda_ss)
            - sp.psi(self.m_W*self.m_lambda0 + self.m_lambda_ss_sum[:, np.newaxis])
        )

    def update_nu(self, subtree, ab, uv, Elogprobw_doc, doc, nu, log_nu):
        Elogpi = self.compute_subtree_Elogpi(subtree, ab, uv)

        # TODO oHDP: only add Elogpi if iter < 3
        log_nu[:,:] = np.repeat(Elogprobw_doc, doc.counts, axis=1).T + Elogpi # N x K
        log_nu[:,[self.tree_index(node) for node in self.tree_iter() if node not in subtree]] = -np.inf
        for n in xrange(doc.total):
            (log_nu[n,:], log_norm) = utils.log_normalize(log_nu[n,:])
        nu[:,:] = np.exp(log_nu)

    def update_uv(self, subtree, nu_sums, uv):
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

    def z_score(self, subtree, ElogV):
        score = 0.0
        for node in self.tree_iter(subtree):
            global_node = subtree[node]
            global_idx = self.tree_index(global_node)
            score += ElogV[0,global_idx]
            for global_s in self.node_left_siblings(global_node):
                global_s_idx = self.tree_index(global_s)
                score += ElogV[1,global_s_idx]
        return score

    def c_score(self, subtree, ab, uv, nu, log_nu, ids):
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

        return np.sum(nu[:,ids] * (log_prob_c[ids][np.newaxis,:] - log_nu[:,ids]))

    def w_score(self, doc, nu, Elogprobw_doc, ids):
        return np.sum(nu[:,ids].T * np.repeat(Elogprobw_doc[ids,:], doc.counts, axis=1))

    def compute_Elogpi(self):
        ids = [self.tree_index(node) for node in self.tree_iter()]
        ElogV = utils.log_beta_expectation(self.m_tau)
        Elogpi = np.zeros(self.m_K)

        for node in self.tree_iter():
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)

                Elogpi[idx] += ElogV[0,p_idx]
                for s in self.node_left_siblings(p):
                    s_idx = self.tree_index(s)
                    Elogpi[idx] += ElogV[1,s_idx]

        return Elogpi

    def compute_logEpi(self):
        ids = [self.tree_index(node) for node in self.tree_iter()]
        logEV = utils.beta_log_expectation(self.m_tau)
        logEpi = np.zeros(self.m_K)

        for node in self.tree_iter():
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)

                logEpi[idx] += logEV[0,p_idx]
                for s in self.node_left_siblings(p):
                    s_idx = self.tree_index(s)
                    logEpi[idx] += logEV[1,s_idx]

        return logEpi

    def compute_subtree_Elogpi(self, subtree, ab, uv):
        Elogpi = np.zeros(self.m_K)
        ids = [self.tree_index(node) for node in self.tree_iter(subtree)]
        ElogV = np.zeros((2, self.m_K))
        ElogV[:,ids] = utils.log_beta_expectation(uv[:,ids])
        ElogU = np.zeros((2, self.m_K))
        ElogU[:,ids] = utils.log_beta_expectation(ab[:,ids])

        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)

                # contributions from switching probabilities
                if idx == p_idx:
                    Elogpi[idx] += ElogU[0,p_idx]
                else:
                    Elogpi[idx] += ElogU[1,p_idx]

                # contributions from stick proportions
                Elogpi[idx] += ElogV[0,p_idx]
                for s in self.node_left_siblings(p):
                    s_idx = self.tree_index(s)
                    Elogpi[idx] += ElogV[1,s_idx]

        return Elogpi

    def compute_subtree_logEpi(self, subtree, ab, uv):
        logEpi = np.zeros(self.m_K)
        ids = [self.tree_index(node) for node in self.tree_iter(subtree)]
        logEV = np.zeros((2, self.m_K))
        logEV[:,ids] = utils.beta_log_expectation(uv[:,ids])
        logEU = np.zeros((2, self.m_K))
        logEU[:,ids] = utils.beta_log_expectation(ab[:,ids])

        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)

                # contributions from switching probabilities
                if idx == p_idx:
                    logEpi[idx] += logEU[0,p_idx]
                else:
                    logEpi[idx] += logEU[1,p_idx]

                # contributions from stick proportions
                logEpi[idx] += logEV[0,p_idx]
                for s in self.node_left_siblings(p):
                    s_idx = self.tree_index(s)
                    logEpi[idx] += logEV[1,s_idx]

        return logEpi

    def doc_e_step(self, doc, ss, ElogV, vocab_to_batch_word_map,
                   batch_to_vocab_word_map, var_converge, max_iter=100,
                   predict_doc=None, save_model=False):
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

        nu = np.zeros((num_tokens, self.m_K))
        log_nu = np.log(nu)
        self.update_nu(subtree, ab, uv, Elogprobw_doc, doc, nu, log_nu)
        nu_sums = np.sum(nu, 0)

        converge = None
        score = None
        old_score = None

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

            # compute score

            score = 0.0

            # E[log p(U | gamma_1, gamma_2)] + H(q(U))
            u_ll = utils.log_sticks_score(ab[:,ab_ids], self.m_gamma1, self.m_gamma2)
            score += u_ll
            logging.debug('Score after U components: %f (+ %f)' % (score, u_ll))

            # E[log p(V | beta)] + H(q(V))
            v_ll = utils.log_sticks_score(uv[:,uv_ids], 1.0, self.m_beta)
            score += v_ll
            logging.debug('Score after V components: %f (+ %f)' % (score, v_ll))

            # E[log p(z | V)] + H(q(z))  (note H(q(z)) = 0)
            z_ll = self.z_score(subtree, ElogV)
            score += z_ll
            logging.debug('Score after z components: %f (+ %f)' % (score, z_ll))

            # E[log p(c | U, V)] + H(q(c))
            c_ll = self.c_score(subtree, ab, uv, nu, log_nu, ids)
            score += c_ll
            logging.debug('Score after c components: %f (+ %f)' % (score, c_ll))

            # E[log p(W | theta, c, z)]
            w_ll = self.w_score(doc, nu, Elogprobw_doc, ids)
            score += w_ll
            logging.debug('Score after W component: %f (+ %f)' % (score, w_ll))

            logging.debug('Score: %f' % score)

            if old_score is not None:
                converge = (score - old_score) / abs(old_score)
                if converge < 0:
                    logging.warning('Score is decreasing')
            old_score = score

            iteration += 1

        # update the suff_stat ss
        global_ids = l2g_idx[ids]
        ss.m_tau_ss[global_ids] += 1
        for n in xrange(num_tokens):
            ss.m_lambda_ss[global_ids, token_batch_ids[n]] += nu[n, ids]

        if save_model:
            # save subtree stats
            self.save_subtree(
                self.subtree_output_files.get('subtree', None),
                doc, subtree, l2g_idx)
            self.save_subtree_Elogpi(
                self.subtree_output_files.get('subtree_Elogpi', None),
                doc, subtree, ids, ab, uv)
            self.save_subtree_logEpi(
                self.subtree_output_files.get('subtree_logEpi', None),
                doc, subtree, ids, ab, uv)
            self.save_subtree_lambda_ss(
                self.subtree_output_files.get('subtree_lambda_ss', None),
                doc, ids, nu_sums)

        if predict_doc is not None:
            logEpi = self.compute_subtree_logEpi(subtree, ab, uv)
            # TODO abstract this?
            logEtheta = (
                np.log(self.m_lambda0 + self.m_lambda_ss)
                - np.log(self.m_W*self.m_lambda0 + self.m_lambda_ss_sum[:,np.newaxis])
            )
            score = np.sum(np.log(np.sum(np.exp(logEpi[ids][:,np.newaxis] + logEtheta[l2g_idx[ids],:][:,predict_doc.words]), 0)) * predict_doc.counts)

        return score

    def node_ancestors(self, node):
        return utils.node_ancestors(node)

    def node_left_siblings(self, node):
        return utils.node_left_siblings(node)

    def tree_iter(self, subtree=None):
        if subtree is None:
            return utils.tree_iter(self.m_trunc)
        else:
            return (n for n in utils.tree_iter(self.m_trunc) if n in subtree)

    def tree_index(self, x):
        return utils.tree_index(x, self.m_trunc_idx_m, self.m_trunc_idx_b)

    def subtree_node_candidates(self, subtree):
        return utils.subtree_node_candidates(self.m_trunc, subtree)

    def select_subtree(self, doc, ElogV, num_tokens):
        # TODO abstract stuff below, like subtree candidate
        # modifications... prone to bugs
        logging.debug('Greedily selecting subtree for ' + str(doc.id))

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

        old_score = 0.0

        ids = [self.tree_index(nod) for nod in self.tree_iter(subtree)]
        logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))
        logging.debug('Subtree global ids: %s'
            % ' '.join(str(l2g_idx[i]) for i in ids))

        Elogprobw_doc = self.m_Elogprobw[l2g_idx, :][:, doc.words]
        nu = np.zeros((num_tokens, self.m_K))
        log_nu = np.log(nu)
        self.update_nu(
            subtree, prior_ab, prior_uv, Elogprobw_doc, doc, nu, log_nu)

        # E[log p(z | V)] + H(q(z))  (note H(q(z)) = 0)
        z_ll = self.z_score(subtree, ElogV)
        old_score += z_ll
        logging.debug('Score after z components: %f (+ %f)'
            % (old_score, z_ll))

        # E[log p(c | U, V)] + H(q(c))
        # TODO is it a bug that the equivalent computation in
        # oHDP does not account for types appearing more than
        # once?  (Uses . rather than ._all .)
        c_ll = self.c_score(subtree, prior_ab, prior_uv, nu, log_nu, ids)
        old_score += c_ll
        logging.debug('Score after c components: %f (+ %f)'
            % (old_score, c_ll))

        # E[log p(W | theta, c, z)]
        w_ll = self.w_score(doc, nu, self.m_Elogprobw[l2g_idx, :][:, doc.words], ids)
        old_score += w_ll
        logging.debug('Score after W component: %f (+ %f)'
            % (old_score, w_ll))

        candidate_nu = nu
        candidate_log_nu = log_nu

        while True:
            best_node = None
            best_global_node = None
            best_score = None

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
                logging.debug('Subtree global ids: %s'
                    % ' '.join(str(l2g_idx[i]) for i in ids))

                Elogprobw_doc = self.m_Elogprobw[l2g_idx, :][:, doc.words]
                self.update_nu(subtree, prior_ab, prior_uv, Elogprobw_doc, doc,
                    candidate_nu, candidate_log_nu)

                candidate_score = 0.0

                # E[log p(z | V)] + H(q(z))  (note H(q(z)) = 0)
                z_ll = self.z_score(subtree, ElogV)
                candidate_score += z_ll
                logging.debug('Score after z components: %f (+ %f)'
                    % (candidate_score, z_ll))

                # E[log p(c | U, V)] + H(q(c))
                # TODO is it a bug that the equivalent computation in
                # oHDP does not account for types appearing more than
                # once?  (Uses . rather than ._all .)
                c_ll = self.c_score(subtree, prior_ab, prior_uv, candidate_nu, candidate_log_nu, ids)
                candidate_score += c_ll
                logging.debug('Score after c components: %f (+ %f)'
                    % (candidate_score, c_ll))

                # E[log p(W | theta, c, z)]
                w_ll = self.w_score(doc, candidate_nu, self.m_Elogprobw[l2g_idx, :][:, doc.words], ids)
                candidate_score += w_ll
                logging.debug('Score after W component: %f (+ %f)'
                    % (candidate_score, w_ll))

                if best_score is None or candidate_score > best_score:
                    best_node = node
                    best_global_node = global_node
                    best_score = candidate_score

                del subtree[node]
                l2g_idx[idx] = 0
                if p in subtree and node[-1] == 0:
                    p_idx = self.tree_index(p)
                    prior_ab[:,p_idx] = [1.0, 0.0]
                if left_s in subtree:
                    left_s_idx = self.tree_index(left_s)
                    prior_uv[:,left_s_idx] = [1.0, 0.0]

            if best_score is None: # no candidates
                break

            converge = (best_score - old_score) / abs(old_score)
            if converge < self.m_delta:
                break

            logging.debug('Selecting global node %s for local node %s'
                % (str(best_global_node), str(best_node)))
            logging.debug('Score: %f' % best_score)

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
            score = best_score

            ids = [self.tree_index(nod) for nod in self.tree_iter(subtree)]
            logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))
            logging.debug('Subtree global ids: %s'
                % ' '.join(str(l2g_idx[i]) for i in ids))

            old_score = score

        logging.debug('Score: %f' % old_score)

        ids = [self.tree_index(nod) for nod in self.tree_iter(subtree)]
        logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))
        logging.debug('Subtree global ids: %s'
            % ' '.join(str(l2g_idx[i]) for i in ids))

        return (subtree, l2g_idx, g2l_idx)

    def update_ss_stochastic(self, ss, batch_to_vocab_word_map):
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

    def save_global(self, output_files):
        self.save_lambda_ss(output_files.get('lambda_ss', None))
        self.save_logEtheta(output_files.get('logEtheta', None))
        self.save_Elogtheta(output_files.get('Elogtheta', None))
        self.save_logEpi(output_files.get('logEpi', None))
        self.save_Elogpi(output_files.get('Elogpi', None))
        self.save_pickle(output_files.get('pickle', None))

    def save_lambda_ss(self, f):
        lambdas = self.m_lambda_ss + self.m_lambda0
        self.save_rows(f, lambdas)

    def save_logEtheta(self, f):
        logEtheta = utils.dirichlet_log_expectation(self.m_lambda0 + self.m_lambda_ss)
        self.save_rows(f, logEtheta)

    def save_Elogtheta(self, f):
        Elogtheta = utils.log_dirichlet_expectation(self.m_lambda0 + self.m_lambda_ss)
        self.save_rows(f, Elogtheta)

    def save_logEpi(self, f):
        logEpi = self.compute_logEpi()
        self.save_rows(f, logEpi[:,np.newaxis])

    def save_Elogpi(self, f):
        Elogpi = self.compute_Elogpi()
        self.save_rows(f, Elogpi[:,np.newaxis])

    def save_pickle(self, f):
        cPickle.dump(self, f, -1)

    def save_subtree_lambda_ss(self, f, doc, ids, doc_lambda_ss):
        self.save_subtree_row(f, doc, doc_lambda_ss[ids] + self.m_lambda0)

    def save_subtree_logEpi(self, f, doc, subtree, ids, ab, uv):
        logEpi = self.compute_subtree_logEpi(subtree, ab, uv)
        self.save_subtree_row(f, doc, logEpi[ids])

    def save_subtree_Elogpi(self, f, doc, subtree, ids, ab, uv):
        Elogpi = self.compute_subtree_Elogpi(subtree, ab, uv)
        self.save_subtree_row(f, doc, Elogpi[ids])

    def save_subtree(self, f, doc, subtree, l2g_idx):
        global_ids = (l2g_idx[self.tree_index(nod)]
                      for nod in self.tree_iter(subtree))
        self.save_subtree_row(f, doc, global_ids)

    def save_rows(self, f, m):
        if f is not None:
            for v in m:
                line = ' '.join([str(x) for x in v])
                f.write('%s\n' % line)

    def save_subtree_row(self, f, doc, v):
        if f is not None:
            line = ' '.join([str(x) for x in v])
            f.write('%s %s\n' % (str(doc.id), line))
