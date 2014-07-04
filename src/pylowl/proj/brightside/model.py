import logging
import numpy as np
import numpy.linalg as la
import numpy.random as nprand
import scipy.special as sp
import itertools as it
import utils
import random
import cPickle


# TODO assert var beta/dirichlet parameters no smaller than prior


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class suff_stats(object):
    def __init__(self, K, Wt, Dt, Ut):
        self.m_batchsize = Dt
        self.m_uv_ss = np.zeros((Ut, K))
        self.m_tau_ss = np.zeros(K)
        self.m_lambda_ss = np.zeros((K, Wt))


class model(object):
    def __init__(self,
                 trunc,
                 D,
                 W,
                 U,
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
                 subtree_f=None,
                 subtree_Elogpi_f=None,
                 subtree_logEpi_f=None,
                 subtree_Elogtheta_f=None,
                 subtree_logEtheta_f=None,
                 subtree_lambda_ss_f=None):

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

        self.m_U = U

        self.m_users = [None] * U
        self.m_r_users = dict()
        self.m_user_subtrees = [None] * U
        self.m_user_l2g_ids = np.zeros((U, self.m_K), dtype=np.uint)
        self.m_user_g2l_ids = np.zeros((U, self.m_K), dtype=np.uint)

        self.m_uv = np.zeros((2, self.m_U, self.m_K))
        self.m_uv[0] = 1.0
        self.m_uv[1] = beta
        self.m_uv_ss = np.zeros((self.m_U, self.m_K))

        self.m_tau = np.zeros((2, self.m_K))
        self.m_tau[0] = 1.0
        self.m_tau[1] = alpha
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
        self.m_num_docs_processed = 0

        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)

        self.subtree_f = subtree_f
        self.subtree_Elogpi_f = subtree_Elogpi_f
        self.subtree_logEpi_f = subtree_logEpi_f
        self.subtree_Elogtheta_f = subtree_Elogtheta_f
        self.subtree_logEtheta_f = subtree_logEtheta_f
        self.subtree_lambda_ss_f = subtree_lambda_ss_f

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

        self.m_lambda_ss = eff_init_samples * init_noise_weight * nprand.dirichlet(100 * np.ones(self.m_W) / float(self.m_W), self.m_K)
        self.m_lambda_ss[:, batch_to_vocab_word_map] += eff_init_samples * (1 - init_noise_weight) * cluster_means

        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)
        self.m_Elogprobw = utils.log_dirichlet_expectation(self.m_lambda0 + self.m_lambda_ss)

    def process_documents(self, docs, var_converge, update=True,
                          predict_docs=None):
        docs = list(docs)
        doc_count = len(docs)

        if predict_docs is None:
            predict_docs = [None] * doc_count

        # Find the unique words in this mini-batch of documents...
        self.m_num_docs_processed += doc_count

        users_to_batch_map = dict()
        batch_to_users_map = []
        docs_new_user = []
        # mapping from word types in this mini-batch to unique
        # consecutive integers
        vocab_to_batch_word_map = dict()
        # list of unique word types, in order of first appearance
        batch_to_vocab_word_map = []
        for doc in docs:
            if doc.user in self.m_r_users:
                doc.user_idx = self.m_r_users[doc.user]
                docs_new_user.append(False)
            else:
                user_idx = len(self.m_r_users)
                self.m_users[user_idx] = doc.user
                self.m_r_users[doc.user] = user_idx
                doc.user_idx = user_idx
                docs_new_user.append(True)

            if doc.user_idx not in users_to_batch_map:
                batch_to_users_map.append(doc.user_idx)
                users_to_batch_map[doc.user_idx] = len(users_to_batch_map)

            for w in doc.words:
                if w not in vocab_to_batch_word_map:
                    vocab_to_batch_word_map[w] = len(vocab_to_batch_word_map)
                    batch_to_vocab_word_map.append(w)

        # number of unique word types in this mini-batch
        num_tokens = sum([sum(doc.counts) for doc in docs])
        Wt = len(batch_to_vocab_word_map)

        logging.info('Processing %d docs spanning %d tokens, %d types'
            % (doc_count, num_tokens, Wt))

        ss = suff_stats(self.m_K, Wt, doc_count, Ut)

        # First row of ElogV is E[log(V)], second row is E[log(1 - V)]
        ids = [self.tree_index(node) for node in self.tree_iter()]
        ElogV = utils.log_beta_expectation(self.m_tau)

        # run variational inference on some new docs
        score = 0.0
        count = 0
        for (doc, predict_doc, new_user) in it.izip(docs, predict_docs, docs_new_user):
            doc_score = self.doc_e_step(doc, ss, ElogV, vocab_to_batch_word_map,
                batch_to_vocab_word_map, users_to_batch_map, batch_to_users_map,
                var_converge, predict_doc=predict_doc, new_user=new_user)

            score += doc_score
            if predict_doc is None:
                count += doc.total
            else:
                count += predict_doc.total

        if update:
            self.update_ss_stochastic(ss, batch_to_vocab_word_map,
                                      batch_to_users_map)
            self.update_lambda()
            self.update_tau()
            self.update_uv()
            self.m_t += 1

        return (score, count, doc_count)

    def update_lambda(self):
        self.m_Elogprobw = (
            sp.psi(self.m_lambda0 + self.m_lambda_ss)
            - sp.psi(self.m_W*self.m_lambda0 + self.m_lambda_ss_sum[:, np.newaxis])
        )

    def update_xi(self, subtree_leaves, uv, Elogprobw_doc, doc, nu, xi, log_xi):
        log_xi[:] = self.compute_subtree_Elogpi(subtree_leaves, uv)
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                p_level = self.node_level(p)
                log_xi[idx] += np.sum(nu[idx,:,p_level] * Elogprobw_doc[p_idx,:] * doc.counts)

        log_xi[[self.tree_index(node) for node in self.tree_iter() if node not in subtree_leaves]] = -np.inf
        (log_xi[:], log_norm) = utils.log_normalize(log_xi)
        xi[:] = np.exp(log_xi)

    def update_nu(self, subtree, subtree_leaves, ab, Elogprobw_doc, doc, xi, nu, log_nu):
        log_nu[:] = -np.inf
        Elogchi = self.compute_subtree_Elogchi(subtree, ab)
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                p_level = self.node_level(p)
                log_nu[idx,:,p_level] = Elogchi[p_idx] + xi[idx] * np.sum(Elogprobw_doc[p_idx,:] * doc.counts)

        (log_nu[:,:], log_norm) = utils.log_normalize(log_nu)
        nu[:,:] = np.exp(log_nu)

    def update_ab(self, subtree_leaves, nu_sums, ab):
        ab[0] = self.m_gamma1 + nu_sums
        ab[1] = self.m_gamma2
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            node_level = self.node_level(node)
            ab[:, idx, node_level] = [1.0, 0.0]
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                p_level = self.node_level(p)
                for pp in self.node_ancestors(p):
                    pp_idx = self.tree_index(pp)
                    pp_level = self.node_level(pp)
                    ab[1, idx, pp_level] += nu_sums[idx, p_level]

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

    def update_uv(self):
        self.m_uv[0] = self.m_uv_ss + 1.0
        self.m_uv[1] = self.m_beta
        for user_idx in xrange(self.m_U):
            subtree = self.m_user_subtrees[user_idx]
            for node in self.tree_iter(subtree):
                idx = self.tree_index(node)
                if node[:-1] + (node[-1] + 1,) not in subtree: # rightmost child
                    self.m_uv[:,user_idx,idx] = [1., 0.]
                for s in self.node_left_siblings(node):
                    s_idx = self.tree_index(s)
                    ss.m_uv[1, user_idx, s_idx] += self.m_uv_ss[user_idx, idx]

    def z_likelihood(self, subtree, ElogV):
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

    def zeta_likelihood(self, subtree_leaves, uv, xi, log_xi, ids_leaves):
        self.check_uv_edge_cases(uv, subtree_leaves, ids_leaves)
        self.check_xi_edge_cases(xi)
        self.check_log_xi_edge_cases(log_xi)
        self.check_subtree_ids(subtree_leaves, ids_leaves)

        Elogpi = self.compute_subtree_Elogpi(subtree_leaves, uv):
        return np.sum(xi[ids_leaves] * (Elogpi[ids_leaves] - log_xi[ids_leaves]))

    def c_likelihood(self, subtree, subtree_leaves, ab, nu, log_nu, ids):
        self.check_ab_edge_cases(ab, subtree, ids)
        self.check_nu_edge_cases(nu)
        self.check_log_nu_edge_cases(log_nu)
        self.check_subtree_ids(subtree, ids)

        likelihood = 0.0
        Elogchi = self.compute_subtree_Elogchi(subtree, ab):
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                p_level = self.node_level(p)
                likelihood += np.sum(nu[idx,:,p_level] * (Elogchi[p_idx] - log_nu[idx,:,p_level]))
        return likelihood

    def w_likelihood(self, doc, nu, xi, Elogprobw_doc):
        self.check_nu_edge_cases(nu)
        self.check_xi_edge_cases(xi)

        likelihood = 0.0
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                p_level = self.node_level(p)
                likelihood += xi[idx] * np.sum(nu[idx,:,p_level] * Elogprobw_doc[p_idx,:] * doc.counts)
        return likelihood

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

    def compute_subtree_Elogpi(self, subtree_leaves, uv):
        Elogpi = np.zeros(self.m_K)
        ids = [self.tree_index(node) for node in self.tree_iter(subtree_leaves)]
        ElogV = np.zeros((2, self.m_K))
        ElogV[:,ids] = utils.log_beta_expectation(uv[:,ids])

        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                Elogpi[idx] += ElogV[0,p_idx]
                for s in self.node_left_siblings(p):
                    s_idx = self.tree_index(s)
                    Elogpi[idx] += ElogV[1,s_idx]

        return Elogpi

    def compute_subtree_logEpi(self, subtree_leaves, uv):
        logEpi = np.zeros(self.m_K)
        ids = [self.tree_index(node) for node in self.tree_iter(subtree_leaves)]
        logEV = np.zeros((2, self.m_K))
        logEV[:,ids] = utils.beta_log_expectation(uv[:,ids])

        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                logEpi[idx] += logEV[0,p_idx]
                for s in self.node_left_siblings(p):
                    s_idx = self.tree_index(s)
                    logEpi[idx] += logEV[1,s_idx]

        return logEpi

    def compute_subtree_Elogchi(self, subtree, ab):
        Elogchi = np.zeros(self.m_K)
        ids = [self.tree_index(node) for node in self.tree_iter(subtree)]
        ElogU = np.zeros((2, self.m_K))
        ElogU[:,ids] = utils.log_beta_expectation(ab[:,ids])

        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                if idx == p_idx:
                    Elogchi[idx] += ElogU[0,p_idx]
                else:
                    Elogchi[idx] += ElogU[1,p_idx]

        return Elogchi

    def compute_subtree_logEchi(self, subtree, ab):
        logEchi = np.zeros(self.m_K)
        ids = [self.tree_index(node) for node in self.tree_iter(subtree)]
        logEU = np.zeros((2, self.m_K))
        logEU[:,ids] = utils.beta_log_expectation(ab[:,ids])

        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                if idx == p_idx:
                    logEchi[idx] += logEU[0,p_idx]
                else:
                    logEchi[idx] += logEU[1,p_idx]

        return logEchi

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
                   batch_to_vocab_word_map, users_to_batch_map,
                   batch_to_users_map, var_converge, max_iter=100,
                   predict_doc=None, new_user=True):

        num_tokens = sum(doc.counts)

        logging.debug('Performing E-step on doc spanning %d tokens, %d types'
            % (num_tokens, len(doc.words)))

        # each position of this list represents a token in our document;
        # the value in that position is the word type id specific to
        # this mini-batch of documents (a unique integer between zero
        # and the number of types in this mini-batch)
        batch_ids = [vocab_to_batch_word_map[w] for w in doc.words]
        token_batch_ids = np.repeat(batch_ids, doc.counts)

        if new_user:
            (subtree, l2g_idx, g2l_idx) = self.select_subtree(doc, ElogV, num_tokens)
            self.m_user_subtrees[doc.user_idx] = subtree
            self.m_user_l2g_ids[doc.user_idx] = l2g_idx
            self.m_user_g2l_ids[doc.user_idx] = g2l_idx
        else:
            subtree = self.m_user_subtrees[doc.user_idx]
            l2g_idx = self.m_user_l2g_ids[doc.user_idx]
            g2l_idx = self.m_user_g2l_ids[doc.user_idx]

        ids = [self.tree_index(node) for node in self.tree_iter(subtree)]

        subtree_leaves = dict((node, subtree[node]) # TODO abstract
                              for node in self.tree_iter(subtree)
                              if node + (0,) not in subtree)
        ids_leaves = [self.tree_index(node)
                      for node in self.tree_iter(subtree_leaves)]

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

        ab = np.zeros((2, self.m_K, self.m_depth))
        ab[0] = 1.0
        ab_leaf_ids = []
        ab_level_ids = []
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in self.node_ancestors(node):
                p_level = self.node_level(p)
                ab[:,idx,p_level] = [self.m_gamma1, self.m_gamma2]
                ab_leaf_ids.append(idx)
                ab_level_ids.append(p_level)

        nu = np.zeros((self.m_K, num_tokens, self.m_depth))
        log_nu = np.log(nu)
        self.update_nu(subtree, subtree_leaves, ab, Elogprobw_doc, doc, xi, nu, log_nu)
        nu_sums = np.sum(nu, 1)

        xi = np.zeros((self.m_K,))
        log_xi = np.log(xi)
        self.update_xi(subtree_leaves, uv, Elogprobw_doc, doc, nu, xi, log_xi)

        converge = None
        likelihood = None
        old_likelihood = None

        iteration = 0
        # not yet support second level optimization yet, to be done in the
        # future
        while iteration < max_iter and (converge is None or converge < 0.0 or converge > var_converge):
            logging.debug('Updating document variational parameters (iteration: %d)' % iteration)
            # update variational parameters

            self.update_nu(subtree, subtree_leaves, ab, Elogprobw_doc, doc, xi, nu, log_nu)
            nu_sums = np.sum(nu, 1)
            self.update_xi(subtree_leaves, uv, Elogprobw_doc, doc, nu, xi, log_xi)
            self.update_ab(subtree_leaves, nu_sums, ab)

            # compute likelihood

            likelihood = 0.0

            # E[log p(U | gamma_1, gamma_2)] + H(q(U))
            u_ll = utils.log_sticks_likelihood(ab[:,ab_leaf_ids,ab_depth_ids], self.m_gamma1, self.m_gamma2)
            likelihood += u_ll
            logging.debug('Log-likelihood after U components: %f (+ %f)' % (likelihood, u_ll))

            # E[log p(V | beta)] + H(q(V))
            v_ll = utils.log_sticks_likelihood(uv[:,uv_ids], 1.0, self.m_beta)
            likelihood += v_ll
            logging.debug('Log-likelihood after V components: %f (+ %f)' % (likelihood, v_ll))

            # E[log p(z | V)] + H(q(z))  (note H(q(z)) = 0)
            z_ll = self.z_likelihood(subtree, ElogV)
            likelihood += z_ll
            logging.debug('Log-likelihood after z components: %f (+ %f)' % (likelihood, z_ll))

            # E[log p(c | U, zeta)] + H(q(c))
            c_ll = self.c_likelihood(subtree, subtree_leaves, ab, nu, log_nu, ids)
            likelihood += c_ll
            logging.debug('Log-likelihood after c components: %f (+ %f)' % (likelihood, c_ll))

            # E[log p(zeta | V)] + H(q(zeta))
            zeta_ll = self.zeta_likelihood(subtree_leaves, uv, xi, log_xi, ids_leaves)
            likelihood += zeta_ll
            logging.debug('Log-likelihood after zeta components: %f (+ %f)'
                % (likelihood, zeta_ll))

            # E[log p(W | theta, c, zeta, z)]
            w_ll = self.w_likelihood(doc, nu, xi, Elogprobw_doc, ids)
            likelihood += w_ll
            logging.debug('Log-likelihood after W component: %f (+ %f)' % (likelihood, w_ll))

            logging.debug('Log-likelihood: %f' % likelihood)

            if old_likelihood is not None:
                converge = (likelihood - old_likelihood) / abs(old_likelihood)
                if converge < 0:
                    logging.warning('Log-likelihood is decreasing')
            old_likelihood = likelihood

            iteration += 1

        if self.subtree_f is not None:
            self.save_subtree(self.subtree_f,
                doc, subtree, l2g_idx)
        if self.subtree_Elogpi_f is not None:
            self.save_subtree_Elogpi(self.subtree_Elogpi_f,
                doc, subtree_leaves, ids, uv)
        if self.subtree_logEpi_f is not None:
            self.save_subtree_logEpi(self.subtree_logEpi_f,
                doc, subtree_leaves, ids, uv)
        if self.subtree_Elogtheta_f is not None:
            self.save_subtree_Elogtheta(self.subtree_Elogtheta_f,
                doc, ids, nu_sums)
        if self.subtree_logEtheta_f is not None:
            self.save_subtree_logEtheta(self.subtree_logEtheta_f,
                doc, ids, nu_sums)
        if self.subtree_lambda_ss_f is not None:
            self.save_subtree_lambda_ss(self.subtree_lambda_ss_f,
                doc, ids, nu_sums)

        # update the suff_stat ss
        global_ids = l2g_idx[ids]
        ss.m_tau_ss[global_ids] += 1
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                p_level = self.node_level(p)
                ss.m_uv_ss[users_to_batch_map[doc.user_idx], p_idx] += xi[idx]
                for n in xrange(num_tokens):
                    ss.m_lambda_ss[l2g_idx[p_idx], token_batch_ids[n]] += nu[idx, n, p_level] * xi[idx]

        if predict_doc is not None:
            logEpi = self.compute_subtree_logEpi(subtree_leaves, uv)
            logEchi = self.compute_subtree_logEchi(subtree, ab)
            # TODO abstract this?
            logEtheta = (
                np.log(self.m_lambda0 + self.m_lambda_ss)
                - np.log(self.m_W*self.m_lambda0 + self.m_lambda_ss_sum[:,np.newaxis])
            )
            logEpichi = np.zeros((self.m_K,))
            for node in self.tree_iter(subtree_leaves):
                idx = self.tree_index(node)
                for p in it.chain((node,), self.node_ancestors(node)):
                    p_idx = self.tree_index(p)
                    p_level = self.node_level(p)
                    logEpichi[p_idx] = logEchi[p_idx] + logEpi[idx]
            likelihood = np.sum(
                np.log(np.sum(
                    np.exp(
                        logEpichi[ids][:,np.newaxis]
                        + logEtheta[l2g_idx[ids],:][:,predict_doc.words]
                    ),
                    0
                ))
                * predict_doc.counts
            )

        return likelihood

    def node_level(self, node):
        return len(node) - 1

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
        logging.debug('Greedily selecting subtree for ' + str(doc.identifier))

        # map from local nodes in subtree to global nodes
        subtree = dict()
        subtree[(0,)] = (0,) # root
        subtree_leaves = subtree.copy()
        # map from local (subtree) node indices to global indices
        l2g_idx = np.zeros(self.m_K, dtype=np.uint)
        # map from global node indices to local (subtree) node indices;
        # unmapped global nodes are left at zero (note that there is no
        # ambiguity here, as the local root only maps to the global
        # root and vice-versa)
        g2l_idx = np.zeros(self.m_K, dtype=np.uint)

        prior_uv = np.zeros((2, self.m_K))
        prior_uv[0] = 1.0

        prior_ab = np.zeros((2, self.m_K, self.m_depth))
        prior_ab[0] = 1.0

        old_likelihood = 0.0

        ids = [self.tree_index(nod) for nod in self.tree_iter(subtree)]
        ids_leaves = [self.tree_index(node)
                      for node in self.tree_iter(subtree_leaves)]
        logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))
        logging.debug('Subtree global ids: %s'
            % ' '.join(str(l2g_idx[i]) for i in ids))

        Elogprobw_doc = self.m_Elogprobw[l2g_idx, :][:, doc.words]
        nu = np.zeros((self.m_K, num_tokens, self.m_depth))
        log_nu = np.log(nu)
        self.update_nu(
            subtree, subtree_leaves, prior_ab, Elogprobw_doc, doc, xi, nu, log_nu)

        xi = np.zeros((self.m_K,))
        log_xi = np.log(xi)
        self.update_xi(
            subtree_leaves, prior_uv, Elogprobw_doc, doc, nu, xi, log_xi)

        # E[log p(z | V)] + H(q(z))  (note H(q(z)) = 0)
        z_ll = self.z_likelihood(subtree, ElogV)
        old_likelihood += z_ll
        logging.debug('Log-likelihood after z components: %f (+ %f)'
            % (old_likelihood, z_ll))

        # E[log p(c | U, zeta)] + H(q(c))
        c_ll = self.c_likelihood(subtree, subtree_leaves, prior_ab, nu, log_nu, ids)
        old_likelihood += c_ll
        logging.debug('Log-likelihood after c components: %f (+ %f)'
            % (old_likelihood, c_ll))

        # E[log p(zeta | V)] + H(q(zeta))
        zeta_ll = self.zeta_likelihood(subtree_leaves, prior_uv, xi, log_xi, ids_leaves)
        old_likelihood += zeta_ll
        logging.debug('Log-likelihood after zeta components: %f (+ %f)'
            % (old_likelihood, zeta_ll))

        # E[log p(W | theta, c, zeta, z)]
        w_ll = self.w_likelihood(doc, nu, xi, self.m_Elogprobw[l2g_idx, :][:, doc.words], ids)
        old_likelihood += w_ll
        logging.debug('Log-likelihood after W component: %f (+ %f)'
            % (old_likelihood, w_ll))

        candidate_nu = np.zeros((self.m_K, num_tokens, self.m_depth))
        candidate_log_nu = np.log(nu)

        candidate_xi = np.zeros((self.m_K,))
        candidate_log_xi = np.log(xi)

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
                subtree_leaves[node] = global_node
                p = node[:-1]
                if p in subtree_leaves:
                    del subtree_leaves[p]
                l2g_idx[idx] = global_idx
                for p in self.node_ancestors(node):
                    p_level = self.node_level(p)
                    prior_ab[:,idx,p_level] = [self.m_gamma1, self.m_gamma2]
                left_s = node[:-1] + (node[-1] - 1,)
                if left_s in subtree:
                    left_s_idx = self.tree_index(left_s)
                    prior_uv[:,left_s_idx] = [1.0, self.m_beta]

                ids = [self.tree_index(nod) for nod in self.tree_iter(subtree)]
                ids_leaves = [self.tree_index(node)
                              for node in self.tree_iter(subtree_leaves)]
                logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))
                logging.debug('Subtree global ids: %s'
                    % ' '.join(str(l2g_idx[i]) for i in ids))

                Elogprobw_doc = self.m_Elogprobw[l2g_idx, :][:, doc.words]
                self.update_nu(subtree, subtree_leaves, prior_ab, Elogprobw_doc, doc,
                    candidate_xi, candidate_nu, candidate_log_nu)

                self.update_xi(subtree_leaves, prior_uv, Elogprobw_doc, doc,
                    candidate_nu, candidate_xi, candidate_log_xi)

                candidate_likelihood = 0.0

                # E[log p(z | V)] + H(q(z))  (note H(q(z)) = 0)
                z_ll = self.z_likelihood(subtree, ElogV)
                candidate_likelihood += z_ll
                logging.debug('Log-likelihood after z components: %f (+ %f)'
                    % (candidate_likelihood, z_ll))

                # E[log p(c | U, zeta)] + H(q(c))
                c_ll = self.c_likelihood(subtree, subtree_leaves, prior_ab, candidate_nu, candidate_log_nu, ids)
                candidate_likelihood += c_ll
                logging.debug('Log-likelihood after c components: %f (+ %f)'
                    % (candidate_likelihood, c_ll))

                # E[log p(zeta | V)] + H(q(zeta))
                zeta_ll = self.zeta_likelihood(subtree_leaves, prior_uv, candidate_xi, candidate_log_xi, ids_leaves)
                candidate_likelihood += zeta_ll
                logging.debug('Log-likelihood after zeta components: %f (+ %f)'
                    % (candidate_likelihood, zeta_ll))

                # E[log p(W | theta, c, zeta, z)]
                w_ll = self.w_likelihood(doc, candidate_nu, candidate_xi, self.m_Elogprobw[l2g_idx, :][:, doc.words], ids)
                candidate_likelihood += w_ll
                logging.debug('Log-likelihood after W component: %f (+ %f)'
                    % (candidate_likelihood, w_ll))

                if best_likelihood is None or candidate_likelihood > best_likelihood:
                    best_node = node
                    best_global_node = global_node
                    best_likelihood = candidate_likelihood

                del subtree[node]
                del subtree_leaves[node]
                p = node[:-1]
                if node[-1] == 0 and p in subtree:
                    subtree_leaves[p] = subtree[p]
                l2g_idx[idx] = 0
                for p in self.node_ancestors(node):
                    p_level = self.node_level(p)
                    prior_ab[:,idx,p_level] = [1.0, 0.0]
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
            subtree_leaves[best_node] = best_global_node
            p = best_node[:-1]
            if p in subtree_leaves:
                del subtree_leaves[p]
            idx = self.tree_index(best_node)
            global_idx = self.tree_index(best_global_node)
            l2g_idx[idx] = global_idx
            g2l_idx[global_idx] = idx
            for p in self.node_ancestors(best_node):
                p_level = self.node_level(p)
                prior_ab[:,idx,p_level] = [self.m_gamma1, self.m_gamma2]
            left_s = best_node[:-1] + (best_node[-1] - 1,)
            if left_s in subtree:
                left_s_idx = self.tree_index(left_s)
                prior_uv[:,left_s_idx] = [1.0, self.m_beta]
            likelihood = best_likelihood

            ids = [self.tree_index(nod) for nod in self.tree_iter(subtree)]
            logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))
            logging.debug('Subtree global ids: %s'
                % ' '.join(str(l2g_idx[i]) for i in ids))

            old_likelihood = likelihood

        logging.debug('Log-likelihood: %f' % old_likelihood)

        ids = [self.tree_index(nod) for nod in self.tree_iter(subtree)]
        logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))
        logging.debug('Subtree global ids: %s'
            % ' '.join(str(l2g_idx[i]) for i in ids))

        return (subtree, l2g_idx, g2l_idx)

    def update_ss_stochastic(self, ss, batch_to_vocab_word_map,
                             batch_to_users_map):
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

        self.m_uv_ss *= (1 - rho)
        self.m_uv_ss[batch_to_users_map, :] += (
            rho * ss.m_uv_ss * self.m_D / ss.m_batchsize
        )

    def save_lambda_ss(self, f):
        lambdas = self.m_lambda_ss + self.m_lambda0
        for row in lambdas:
            line = ' '.join([str(x) for x in row])
            f.write(line + '\n')

    def save_logEtheta(self, f):
        logEtheta = utils.dirichlet_log_expectation(self.m_lambda0 + self.m_lambda_ss)
        for row in logEtheta:
            line = ' '.join([str(x) for x in row])
            f.write(line + '\n')

    def save_Elogtheta(self, f):
        Elogtheta = utils.log_dirichlet_expectation(self.m_lambda0 + self.m_lambda_ss)
        for row in Elogtheta:
            line = ' '.join([str(x) for x in row])
            f.write(line + '\n')

    def save_logEpi(self, f):
        logEpi = self.compute_logEpi().T
        f.write('\n'.join(str(x) for x in logEpi))
        f.write('\n')

    def save_Elogpi(self, f):
        Elogpi = self.compute_Elogpi().T
        f.write('\n'.join(str(x) for x in Elogpi))
        f.write('\n')

    def save_subtree_lambda_ss(self, f, doc, ids, nu_sums):
        # TODO lambda0?
        f.write(str(doc.identifier) + ' ')
        f.write(' '.join(str(x) for x in nu_sums[ids]))
        f.write('\n')

    def save_subtree_logEtheta(self, f, doc, ids, nu_sums):
        logEtheta = utils.dirichlet_log_expectation(self.m_lambda0 + nu_sums)
        f.write(str(doc.identifier) + ' ')
        f.write(' '.join(str(logEtheta[i]) for i in ids))
        f.write('\n')

    def save_subtree_Elogtheta(self, f, doc, ids, nu_sums):
        Elogtheta = utils.log_dirichlet_expectation(self.m_lambda0 + nu_sums)
        f.write(str(doc.identifier) + ' ')
        f.write(' '.join(str(Elogtheta[i]) for i in ids))
        f.write('\n')

    def save_subtree_logEpi(self, f, doc, subtree_leaves, ids, uv):
        logEpi = self.compute_subtree_logEpi(subtree_leaves, uv)
        f.write(str(doc.identifier) + ' ')
        f.write(' '.join(str(logEpi[i]) for i in ids))
        f.write('\n')

    def save_subtree_Elogpi(self, f, doc, subtree_leaves, ids, uv):
        Elogpi = self.compute_subtree_Elogpi(subtree_leaves, uv)
        f.write(str(doc.identifier) + ' ')
        f.write(' '.join(str(Elogpi[i]) for i in ids))
        f.write('\n')

    def save_subtree(self, f, doc, subtree, l2g_idx):
        f.write(str(doc.identifier) + ' ')
        f.write(' '.join(str(l2g_idx[self.tree_index(nod)])
                         for nod in self.tree_iter(subtree)))
        f.write('\n')

    def save_model(self, f):
        cPickle.dump(self, f, -1)
