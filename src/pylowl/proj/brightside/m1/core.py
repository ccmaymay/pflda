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
    def __init__(self, K, Wt, Dt, Ut, DperUt):
        self.m_batch_D = Dt
        self.m_batch_U = Ut
        self.m_batch_DperU = DperUt
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
                 user_subtree_selection_interval=10,
                 user_doc_reservoir_capacity=10):
        if trunc[0] != 1:
            raise ValueError('Top-level truncation must be one.')

        self.m_trunc = trunc
        self.m_trunc_idx_b = utils.tree_index_b(trunc)
        self.m_trunc_idx_m = utils.tree_index_m(trunc)

        self.m_K = int(np.sum(self.m_trunc_idx_b)) # total no. nodes
        self.m_W = W
        self.m_D = D
        self.m_DperU = D / float(U)
        self.m_depth = len(trunc)
        self.m_beta = beta
        self.m_alpha = alpha
        self.m_gamma1 = gamma1
        self.m_gamma2 = gamma2

        self.m_U = U

        self.m_user_subtree_selection_interval = user_subtree_selection_interval
        self.m_user_subtree_selection_counters = [0] * U
        self.m_user_docs_counts = [0] * U
        self.m_user_docs = [[None] * user_doc_reservoir_capacity for i in xrange(U)]
        self.m_users = [None] * U
        self.m_r_users = dict()
        self.m_user_subtrees = [None] * U
        self.m_user_l2g_ids = np.zeros((U, self.m_K), dtype=np.uint)
        self.m_user_g2l_ids = np.zeros((U, self.m_K), dtype=np.uint)
        self.m_user_lambda_ss_sums = np.zeros((U, self.m_K))

        self.m_uv = np.zeros((2, self.m_U, self.m_K))
        self.m_uv[0] = 1.0
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
                          predict_docs=None):
        docs = list(docs)
        doc_count = len(docs)

        if predict_docs is None:
            predict_docs = [None] * doc_count

        # Find the unique words in this mini-batch of documents...
        self.m_num_docs_processed += doc_count

        users_to_batch_map = dict()
        batch_to_users_map = []
        DperUt = []
        docs_new_user = []
        # mapping from word types in this mini-batch to unique
        # consecutive integers
        vocab_to_batch_word_map = dict()
        # list of unique word types, in order of first appearance
        batch_to_vocab_word_map = []
        for doc in docs:
            if doc.attrs['user'] in self.m_r_users:
                doc.user_idx = self.m_r_users[doc.attrs['user']]
                docs_new_user.append(False)
            else:
                user_idx = len(self.m_r_users)
                self.m_users[user_idx] = doc.attrs['user']
                self.m_r_users[doc.attrs['user']] = user_idx
                doc.user_idx = user_idx
                docs_new_user.append(True)

            utils.reservoir_insert(self.m_user_docs[doc.user_idx], self.m_user_docs_counts[doc.user_idx], doc)
            self.m_user_docs_counts[doc.user_idx] += 1

            if doc.user_idx not in users_to_batch_map:
                batch_to_users_map.append(doc.user_idx)
                users_to_batch_map[doc.user_idx] = len(users_to_batch_map)
                DperUt.append(0)
            DperUt[users_to_batch_map[doc.user_idx]] += 1

            for w in doc.words:
                if w not in vocab_to_batch_word_map:
                    vocab_to_batch_word_map[w] = len(vocab_to_batch_word_map)
                    batch_to_vocab_word_map.append(w)

        Ut = len(batch_to_users_map)

        # number of unique word types in this mini-batch
        num_tokens = sum([sum(doc.counts) for doc in docs])
        Wt = len(batch_to_vocab_word_map)

        logging.info('Processing %d docs spanning %d tokens, %d types'
            % (doc_count, num_tokens, Wt))

        ss = suff_stats(self.m_K, Wt, doc_count, Ut, DperUt)

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

        # update tau ss
        for user_batch_idx in xrange(ss.m_batch_U):
            user_idx = batch_to_users_map[user_batch_idx]
            subtree = self.m_user_subtrees[user_idx]
            ids = [self.tree_index(node) for node in self.tree_iter(subtree)]
            l2g_idx = self.m_user_l2g_ids[user_idx]
            ss.m_tau_ss[l2g_idx[ids]] += 1

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

    def update_xi(self, subtree_leaves, ids, Elogprobw_doc, doc, nu, xi, log_xi, l2g_idx):
        log_xi[:] = self.compute_subtree_Elogpi(subtree_leaves, ids, doc.user_idx, l2g_idx)
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                p_level = self.node_level(p)
                log_xi[idx] += np.sum(nu[idx,:,p_level] * np.repeat(Elogprobw_doc[p_idx,:], doc.counts))

        log_xi[[self.tree_index(node) for node in self.tree_iter() if node not in subtree_leaves]] = -np.inf
        (log_xi[:], log_norm) = utils.log_normalize(log_xi)
        xi[:] = np.exp(log_xi)

    def update_nu(self, subtree, subtree_leaves, ab, Elogprobw_doc, doc, xi, nu, log_nu):
        log_nu[:] = -np.inf
        Elogchi = self.compute_subtree_Elogchi(subtree_leaves, ab)
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            node_level = self.node_level(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                p_level = self.node_level(p)
                log_nu[idx,:,p_level] = Elogchi[idx,p_level] + xi[idx] * np.repeat(Elogprobw_doc[p_idx,:], doc.counts)
            for n in xrange(doc.total):
                (log_nu[idx,n,:node_level+1], log_norm) = utils.log_normalize(log_nu[idx,n,:node_level+1])

        nu[:] = np.exp(log_nu)

    def update_ab(self, subtree_leaves, nu_sums, ab):
        ab[0] = self.m_gamma1 + nu_sums
        ab[1] = self.m_gamma2
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            node_level = self.node_level(node)
            ab[:, idx, node_level] = [1.0, 0.0]
            for p in self.node_ancestors(node):
                p_level = self.node_level(p)
                ab[1, idx, p_level] += np.sum(nu_sums[idx, p_level+1:node_level+1])

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
            if subtree is None:
                continue

            l2g_idx = self.m_user_l2g_ids[user_idx]
            for node in self.tree_iter(subtree):
                idx = self.tree_index(node)
                if node[:-1] + (node[-1] + 1,) not in subtree: # rightmost child
                    self.m_uv[:,user_idx,idx] = [1., 0.]
                for s in self.node_left_siblings(node):
                    s_idx = self.tree_index(s)
                    self.m_uv[1, user_idx, s_idx] += self.m_uv_ss[user_idx, idx]

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

    def zeta_likelihood(self, subtree_leaves, ids_leaves, ids, doc, xi, log_xi, l2g_idx):
        self.check_xi_edge_cases(xi, ids_leaves)
        self.check_xi_edge_cases(np.exp(log_xi), ids_leaves)
        self.check_subtree_ids(subtree_leaves, ids_leaves)

        Elogpi = self.compute_subtree_Elogpi(subtree_leaves, ids, doc.user_idx, l2g_idx)
        return np.sum(xi[ids_leaves] * (Elogpi[ids_leaves] - log_xi[ids_leaves]))

    def c_likelihood(self, subtree, subtree_leaves, ab, nu, log_nu, ids):
        self.check_ab_edge_cases(ab, subtree_leaves)
        self.check_nu_edge_cases(nu, subtree_leaves)
        self.check_nu_edge_cases(np.exp(log_nu), subtree_leaves)
        self.check_subtree_ids(subtree, ids)

        likelihood = 0.0
        Elogchi = self.compute_subtree_Elogchi(subtree_leaves, ab)
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                p_level = self.node_level(p)
                likelihood += np.sum(nu[idx,:,p_level] * (Elogchi[idx,p_level] - log_nu[idx,:,p_level]))
        return likelihood

    def w_likelihood(self, doc, nu, xi, Elogprobw_doc, subtree_leaves, ids_leaves):
        self.check_nu_edge_cases(nu, subtree_leaves)
        self.check_xi_edge_cases(xi, ids_leaves)

        likelihood = 0.0
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                p_level = self.node_level(p)
                likelihood += xi[idx] * np.sum(nu[idx,:,p_level] * np.repeat(Elogprobw_doc[p_idx,:], doc.counts))
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

    def compute_subtree_Elogpi(self, subtree_leaves, ids, user_idx, l2g_idx):
        Elogpi = np.zeros(self.m_K)
        ElogV = np.zeros((2, self.m_K))
        ElogV[:,ids] = utils.log_beta_expectation(self.m_uv[:,user_idx,ids])

        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                Elogpi[idx] += ElogV[0,p_idx]
                for s in self.node_left_siblings(p):
                    s_idx = self.tree_index(s)
                    Elogpi[idx] += ElogV[1,s_idx]

        return Elogpi

    def compute_subtree_logEpi(self, subtree_leaves, ids, user_idx, l2g_idx):
        logEpi = np.zeros(self.m_K)
        logEV = np.zeros((2, self.m_K))
        logEV[:,ids] = utils.beta_log_expectation(self.m_uv[:,user_idx,ids])

        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                logEpi[idx] += logEV[0,p_idx]
                for s in self.node_left_siblings(p):
                    s_idx = self.tree_index(s)
                    logEpi[idx] += logEV[1,s_idx]

        return logEpi

    def compute_subtree_Elogchi(self, subtree_leaves, ab):
        Elogchi = np.zeros((self.m_K,self.m_depth))

        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_level = self.node_level(p)
                ElogU = utils.log_beta_expectation(ab[:,idx,:][:,:p_level+1])
                Elogchi[idx,p_level] = ElogU[0,p_level] + np.sum(ElogU[1,:p_level])

        return Elogchi

    def compute_subtree_logEchi(self, subtree_leaves, ab):
        logEchi = np.zeros((self.m_K,self.m_depth))

        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_level = self.node_level(p)
                logEU = utils.log_beta_expectation(ab[:,idx,:][:,:p_level+1])
                logEchi[idx,p_level] = logEU[0,p_level] + np.sum(logEU[1,:p_level])

        return logEchi

    def check_subtree_ids(self, subtree, ids):
        ids_in_subtree = set(ids)
        for node in self.tree_iter(subtree):
            idx = self.tree_index(node)
            assert idx in ids_in_subtree, 'id %d in subtree but not in id list' % idx
            ids_in_subtree.remove(idx)
        assert not ids_in_subtree, 'ids in id list but not in subtree: %s' % str(ids_in_subtree)

    def check_ab_edge_cases(self, ab, subtree_leaves):
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            node_level = self.node_level(node)
            assert ab[0, idx, node_level] == 1. and ab[1, idx, node_level] == 0., 'leaf %s has ab = %s (require [1, 0])' % (str(node), str(ab[:, idx, node_level]))

    def check_ElogV_edge_cases(self, ElogV):
        for node in self.tree_iter():
            idx = self.tree_index(node)
            if node[-1] + 1 == self.m_trunc[len(node)-1]: # node is last child of its parent in global tree
                assert ElogV[0, idx] == 0. and ElogV[1, idx] == np.inf, 'right-most child %s has ElogV = %s (require [0, inf])' % (str(node), str(ElogV[:, idx]))

    def check_xi_edge_cases(self, xi, ids_leaves):
        assert np.sum(xi[ids_leaves]) - 1 < 1e-9, 'xi does not sum to one: %s' % str(xi)

    def check_nu_edge_cases(self, nu, subtree_leaves):
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            node_level = self.node_level(node)
            nu_row = nu[idx,:,:node_level+1]
            assert not np.isnan(nu_row).any(), 'not all rows of nu are nan-free: %s' % str(nu_row)
            assert la.norm(np.sum(nu_row,1) - 1, np.inf) < 1e-9, 'not all rows of nu sum to one: %s' % str(nu_row)

    def doc_e_step(self, doc, ss, ElogV, vocab_to_batch_word_map,
                   batch_to_vocab_word_map, users_to_batch_map,
                   batch_to_users_map, var_converge, max_iter=100,
                   predict_doc=None, new_user=True):
        user_idx = doc.user_idx
        num_tokens = sum(doc.counts)

        logging.debug('Performing E-step on doc spanning %d tokens, %d types'
            % (num_tokens, len(doc.words)))

        # each position of this list represents a token in our document;
        # the value in that position is the word type id specific to
        # this mini-batch of documents (a unique integer between zero
        # and the number of types in this mini-batch)
        batch_ids = [vocab_to_batch_word_map[w] for w in doc.words]
        token_batch_ids = np.repeat(batch_ids, doc.counts)

        if new_user or self.m_user_subtree_selection_counters[user_idx] % self.m_user_subtree_selection_interval == 0:
            (subtree, l2g_idx, g2l_idx) = self.select_subtree(user_idx, ElogV)
            self.m_user_subtrees[user_idx] = subtree
            self.m_user_l2g_ids[user_idx] = l2g_idx
            self.m_user_g2l_ids[user_idx] = g2l_idx
        else:
            subtree = self.m_user_subtrees[user_idx]
            l2g_idx = self.m_user_l2g_ids[user_idx]
            g2l_idx = self.m_user_g2l_ids[user_idx]
        self.m_user_subtree_selection_counters[user_idx] += 1

        ids = [self.tree_index(node) for node in self.tree_iter(subtree)]

        subtree_leaves = dict((node, subtree[node]) # TODO abstract
                              for node in self.tree_iter(subtree)
                              if node + (0,) not in subtree)
        ids_leaves = [self.tree_index(node)
                      for node in self.tree_iter(subtree_leaves)]

        Elogprobw_doc = self.m_Elogprobw[l2g_idx, :][:, doc.words]

        logging.debug('Initializing document variational parameters')

        uv_ids = []
        for node in self.tree_iter(subtree):
            s = node[:-1] + (node[-1] + 1,) # right sibling
            if s in subtree: # node is not last child of its parent in subtree
                idx = self.tree_index(node)
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

        xi = np.zeros((self.m_K,))
        xi[ids_leaves] = 1./len(ids_leaves)
        log_xi = np.log(xi)

        nu = np.zeros((self.m_K, num_tokens, self.m_depth))
        log_nu = np.log(nu)
        self.update_nu(subtree, subtree_leaves, ab, Elogprobw_doc, doc, xi, nu, log_nu)
        nu_sums = np.sum(nu, 1)

        converge = None
        likelihood = None
        old_likelihood = None

        iteration = 0
        # not yet support second level optimization yet, to be done in the
        # future
        while iteration < max_iter and (converge is None or converge < 0.0 or converge > var_converge):
            logging.debug('Updating document variational parameters (iteration: %d)' % iteration)
            # update variational parameters

            self.update_xi(subtree_leaves, ids, Elogprobw_doc, doc, nu, xi, log_xi, l2g_idx)
            self.update_nu(subtree, subtree_leaves, ab, Elogprobw_doc, doc, xi, nu, log_nu)
            nu_sums = np.sum(nu, 1)
            self.update_ab(subtree_leaves, nu_sums, ab)

            # compute likelihood

            likelihood = 0.0

            # E[log p(U | gamma_1, gamma_2)] + H(q(U))
            u_ll = utils.log_sticks_likelihood(ab[:,ab_leaf_ids,ab_level_ids], self.m_gamma1, self.m_gamma2)
            likelihood += u_ll
            logging.debug('Log-likelihood after U components: %f (+ %f)' % (likelihood, u_ll))

            # E[log p(V | beta)] + H(q(V))
            v_ll = utils.log_sticks_likelihood(self.m_uv[:,user_idx,uv_ids], 1.0, self.m_beta)
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
            zeta_ll = self.zeta_likelihood(subtree_leaves, ids_leaves, ids, doc, xi, log_xi, l2g_idx)
            likelihood += zeta_ll
            logging.debug('Log-likelihood after zeta components: %f (+ %f)'
                % (likelihood, zeta_ll))

            # E[log p(W | theta, c, zeta, z)]
            w_ll = self.w_likelihood(doc, nu, xi, Elogprobw_doc, subtree_leaves, ids_leaves)
            likelihood += w_ll
            logging.debug('Log-likelihood after W component: %f (+ %f)' % (likelihood, w_ll))

            logging.debug('Log-likelihood: %f' % likelihood)

            if old_likelihood is not None:
                converge = (likelihood - old_likelihood) / abs(old_likelihood)
                if converge < 0:
                    logging.warning('Log-likelihood is decreasing')
            old_likelihood = likelihood

            iteration += 1

        # update ss
        for node in self.tree_iter(subtree_leaves):
            idx = self.tree_index(node)
            for p in it.chain((node,), self.node_ancestors(node)):
                p_idx = self.tree_index(p)
                p_level = self.node_level(p)
                ss.m_uv_ss[users_to_batch_map[user_idx], p_idx] += xi[idx]
                for n in xrange(num_tokens):
                    ss.m_lambda_ss[l2g_idx[p_idx], token_batch_ids[n]] += nu[idx, n, p_level] * xi[idx]
                # TODO not consistent: rho...
                self.m_user_lambda_ss_sums[user_idx,l2g_idx[p_idx]] += np.sum(nu[idx, :, p_level]) * xi[idx]

        if predict_doc is not None:
            logEpi = self.compute_subtree_logEpi(subtree_leaves, ids, user_idx, l2g_idx)
            logEchi = self.compute_subtree_logEchi(subtree_leaves, ab)
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
                    logEpichi[p_idx] = logEchi[idx,p_level] + logEpi[idx]
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

    def select_subtree(self, user_idx, ElogV):
        # TODO abstract stuff below, like subtree candidate
        # modifications... prone to bugs
        logging.debug('Greedily selecting subtree for %s' % str(self.m_users[user_idx]))

        docs = self.m_user_docs[user_idx][:self.m_user_docs_counts[user_idx]]
        num_docs = len(docs)

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

        # q(V_{i,j}^{(d)} = 1) = 1 for j+1 = trunc[\ell] (\ell is depth)
        self.m_uv[:,user_idx,:] = [[1.], [0.]]
        self.m_uv_ss[:] = 0.

        prior_ab = np.zeros((2, self.m_K, self.m_depth))
        prior_ab[0] = 1.0

        old_likelihood = 0.0

        ids = [self.tree_index(n) for n in self.tree_iter(subtree)]
        ids_leaves = [self.tree_index(n)
                      for n in self.tree_iter(subtree_leaves)]
        logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))
        logging.debug('Subtree global ids: %s'
            % ' '.join(str(l2g_idx[i]) for i in ids))

        # E[log p(z | V)] + H(q(z))  (note H(q(z)) = 0)
        z_ll = self.z_likelihood(subtree, ElogV)
        old_likelihood += z_ll
        logging.debug('Log-likelihood after z components: %f (+ %f)'
            % (old_likelihood, z_ll))

        for (doc_idx, doc) in enumerate(docs):
            Elogprobw_doc = self.m_Elogprobw[l2g_idx, :][:, doc.words]

            xi = np.zeros((self.m_K,))
            xi[ids_leaves] = 1./len(ids_leaves)
            log_xi = np.log(xi)

            nu = np.zeros((self.m_K, doc.total, self.m_depth))
            log_nu = np.log(nu)

            self.update_nu(
                subtree, subtree_leaves, prior_ab, Elogprobw_doc, doc, xi, nu, log_nu)

            # E[log p(c | U, zeta)] + H(q(c))
            c_ll = self.c_likelihood(subtree, subtree_leaves, prior_ab, nu, log_nu, ids)
            old_likelihood += c_ll
            logging.debug('Log-likelihood after c components: %f (+ %f)'
                % (old_likelihood, c_ll))

            # E[log p(zeta | V)] + H(q(zeta))
            zeta_ll = self.zeta_likelihood(subtree_leaves, ids_leaves, ids, doc, xi, log_xi, l2g_idx)
            old_likelihood += zeta_ll
            logging.debug('Log-likelihood after zeta components: %f (+ %f)'
                % (old_likelihood, zeta_ll))

            # E[log p(W | theta, c, zeta, z)]
            w_ll = self.w_likelihood(doc, nu, xi, Elogprobw_doc, subtree_leaves, ids_leaves)
            old_likelihood += w_ll
            logging.debug('Log-likelihood after W component: %f (+ %f)'
                % (old_likelihood, w_ll))

        while True:
            best_node = None
            best_global_node = None
            best_likelihood = None

            for (node, global_node) in self.subtree_node_candidates(subtree):
                logging.debug('Candidate: global node %s for local node %s'
                    % (str(global_node), str(node)))
                idx = self.tree_index(node)
                global_idx = self.tree_index(global_node)

                p = node[:-1]
                if p in subtree_leaves:
                    del subtree_leaves[p]
                node_level = self.node_level(node)
                prior_ab[:,idx,:node_level] = [[self.m_gamma1], [self.m_gamma2]]
                l2g_idx[idx] = global_idx
                left_s = node[:-1] + (node[-1] - 1,)
                if left_s in subtree:
                    left_s_idx = self.tree_index(left_s)
                    self.m_uv[:,user_idx,left_s_idx] = [1.0, self.m_beta]
                subtree[node] = global_node
                subtree_leaves[node] = global_node

                ids = [self.tree_index(n) for n in self.tree_iter(subtree)]
                ids_leaves = [self.tree_index(n)
                              for n in self.tree_iter(subtree_leaves)]
                logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))
                logging.debug('Subtree global ids: %s'
                    % ' '.join(str(l2g_idx[i]) for i in ids))

                candidate_likelihood = 0.0

                # E[log p(z | V)] + H(q(z))  (note H(q(z)) = 0)
                z_ll = self.z_likelihood(subtree, ElogV)
                candidate_likelihood += z_ll
                logging.debug('Log-likelihood after z components: %f (+ %f)'
                    % (candidate_likelihood, z_ll))

                for (doc_idx, doc) in enumerate(docs):
                    Elogprobw_doc = self.m_Elogprobw[l2g_idx, :][:, doc.words]

                    candidate_xi = np.zeros((self.m_K,))
                    candidate_xi[ids_leaves] = 1./len(ids_leaves)
                    candidate_log_xi = np.log(candidate_xi)

                    candidate_nu = np.zeros((self.m_K, doc.total, self.m_depth))
                    candidate_log_nu = np.log(candidate_nu)

                    self.update_nu(subtree, subtree_leaves, prior_ab, Elogprobw_doc, doc,
                        candidate_xi, candidate_nu, candidate_log_nu)

                    self.update_xi(subtree_leaves, ids, Elogprobw_doc, doc,
                        candidate_nu, candidate_xi, candidate_log_xi, l2g_idx)
                    self.update_nu(subtree, subtree_leaves, prior_ab, Elogprobw_doc, doc,
                        candidate_xi, candidate_nu, candidate_log_nu)

                    # E[log p(c | U, zeta)] + H(q(c))
                    c_ll = self.c_likelihood(subtree, subtree_leaves, prior_ab, candidate_nu, candidate_log_nu, ids)
                    candidate_likelihood += c_ll
                    logging.debug('Log-likelihood after c components: %f (+ %f)'
                        % (candidate_likelihood, c_ll))

                    # E[log p(zeta | V)] + H(q(zeta))
                    zeta_ll = self.zeta_likelihood(subtree_leaves, ids_leaves, ids, doc, candidate_xi, candidate_log_xi, l2g_idx)
                    candidate_likelihood += zeta_ll
                    logging.debug('Log-likelihood after zeta components: %f (+ %f)'
                        % (candidate_likelihood, zeta_ll))

                    # E[log p(W | theta, c, zeta, z)]
                    w_ll = self.w_likelihood(doc, candidate_nu, candidate_xi, Elogprobw_doc, subtree_leaves, ids_leaves)
                    candidate_likelihood += w_ll
                    logging.debug('Log-likelihood after W component: %f (+ %f)'
                        % (candidate_likelihood, w_ll))

                if best_likelihood is None or candidate_likelihood > best_likelihood:
                    best_node = node
                    best_global_node = global_node
                    best_likelihood = candidate_likelihood

                p = node[:-1]
                if node[-1] == 0 and p in subtree:
                    subtree_leaves[p] = subtree[p]
                prior_ab[:,idx,:node_level] = [[1.], [0.]]
                l2g_idx[idx] = 0
                if left_s in subtree:
                    left_s_idx = self.tree_index(left_s)
                    self.m_uv[:,user_idx,left_s_idx] = [1.0, 0.0]
                del subtree[node]
                del subtree_leaves[node]

            if best_likelihood is None: # no candidates
                break

            converge = (best_likelihood - old_likelihood) / abs(old_likelihood)
            if converge < self.m_delta:
                break

            node = best_node
            global_node = best_global_node
            likelihood = best_likelihood

            logging.debug('Selecting global node %s for local node %s'
                % (str(global_node), str(node)))
            logging.debug('Log-likelihood: %f' % likelihood)

            idx = self.tree_index(node)
            global_idx = self.tree_index(global_node)
            p = node[:-1]
            if p in subtree_leaves:
                del subtree_leaves[p]
            node_level = self.node_level(node)
            prior_ab[:,idx,:node_level] = [[self.m_gamma1], [self.m_gamma2]]
            l2g_idx[idx] = global_idx
            g2l_idx[global_idx] = idx
            left_s = node[:-1] + (node[-1] - 1,)
            if left_s in subtree:
                left_s_idx = self.tree_index(left_s)
                self.m_uv[:,user_idx,left_s_idx] = [1.0, self.m_beta]
            subtree[node] = global_node
            subtree_leaves[node] = global_node

            ids = [self.tree_index(n) for n in self.tree_iter(subtree)]
            logging.debug('Subtree ids: %s' % ' '.join(str(i) for i in ids))
            logging.debug('Subtree global ids: %s'
                % ' '.join(str(l2g_idx[i]) for i in ids))

            old_likelihood = likelihood

        logging.debug('Selecting subtree:')
        logging.debug('Log-likelihood: %f' % old_likelihood)

        ids = [self.tree_index(n) for n in self.tree_iter(subtree)]
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
            rho * ss.m_lambda_ss * self.m_D / float(ss.m_batch_D)
        )
        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)

        self.m_tau_ss = (
            (1.0 - rho) * self.m_tau_ss
            + rho * ss.m_tau_ss * self.m_U / float(ss.m_batch_U)
        )

        self.m_uv_ss *= (1 - rho)
        for user_batch_idx in xrange(ss.m_batch_U):
            self.m_uv_ss[batch_to_users_map[user_batch_idx], :] += (
                rho * ss.m_uv_ss[user_batch_idx] * self.m_DperU / float(ss.m_batch_DperU[user_batch_idx])
            )

    def save(self, output_files):
        self.save_lambda_ss(output_files.get('lambda_ss', None))
        self.save_logEtheta(output_files.get('logEtheta', None))
        self.save_Elogtheta(output_files.get('Elogtheta', None))
        self.save_logEpi(output_files.get('logEpi', None))
        self.save_Elogpi(output_files.get('Elogpi', None))
        self.save_pickle(output_files.get('pickle', None))

        # save subtree stats
        for user_idx in xrange(self.m_U):
            subtree = self.m_user_subtrees[user_idx]
            if subtree is None:
                continue

            l2g_idx = self.m_user_l2g_ids[user_idx]
            g2l_idx = self.m_user_g2l_ids[user_idx]

            ids = [self.tree_index(node) for node in self.tree_iter(subtree)]

            subtree_leaves = dict((node, subtree[node]) # TODO abstract
                                  for node in self.tree_iter(subtree)
                                  if node + (0,) not in subtree)
            ids_leaves = [self.tree_index(node)
                          for node in self.tree_iter(subtree_leaves)]

            self.save_subtree(
                output_files.get('subtree', None),
                user_idx, subtree, l2g_idx)
            self.save_subtree_Elogpi(
                output_files.get('subtree_Elogpi', None),
                user_idx, subtree_leaves, ids, l2g_idx)
            self.save_subtree_logEpi(
                output_files.get('subtree_logEpi', None),
                user_idx, subtree_leaves, ids, l2g_idx)
            self.save_subtree_lambda_ss(
                output_files.get('subtree_lambda_ss', None),
                user_idx, ids, self.m_user_lambda_ss_sums[user_idx,l2g_idx])

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

    def save_subtree_lambda_ss(self, f, user_idx, ids, user_lambda_ss):
        self.save_subtree_row(f, user_idx, user_lambda_ss[ids] + self.m_lambda0)

    def save_subtree_logEpi(self, f, user_idx, subtree_leaves, ids, l2g_idx):
        logEpi = self.compute_subtree_logEpi(subtree_leaves, ids, user_idx, l2g_idx)
        self.save_subtree_row(f, user_idx, logEpi[ids])

    def save_subtree_Elogpi(self, f, user_idx, subtree_leaves, ids, l2g_idx):
        Elogpi = self.compute_subtree_Elogpi(subtree_leaves, ids, user_idx, l2g_idx)
        self.save_subtree_row(f, user_idx, Elogpi[ids])

    def save_subtree(self, f, user_idx, subtree, l2g_idx):
        global_ids = (l2g_idx[self.tree_index(nod)]
                      for nod in self.tree_iter(subtree))
        self.save_subtree_row(f, user_idx, global_ids)

    def save_rows(self, f, m):
        if f is not None:
            for v in m:
                line = ' '.join([str(x) for x in v])
                f.write('%s\n' % line)

    def save_subtree_row(self, f, user_idx, v):
        if f is not None:
            line = ' '.join([str(x) for x in v])
            f.write('%s %s\n' % (str(self.m_users[user_idx]), line))
