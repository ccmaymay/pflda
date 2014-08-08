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
                 K,
                 L,
                 D,
                 W,
                 lambda0=0.01,
                 beta=1.,
                 alpha=1.,
                 kappa=0.5,
                 iota=1.,
                 scale=1.,
                 rho_bound=0.,
                 optimal_order=False,
                 sublist_output_files=None):
        self.m_K = K
        self.m_L = L
        self.m_W = W
        self.m_D = D
        self.m_beta = beta
        self.m_alpha = alpha

        self.m_tau = np.zeros((2, self.m_K))
        self.m_tau[0] = 1.0
        self.m_tau[1] = alpha
        self.m_tau[1,self.m_K-1] = 0.

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
        self.m_scale = scale
        self.m_rho_bound = rho_bound
        self.m_t = 0
        self.m_num_docs_processed = 0

        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)

        if sublist_output_files is None:
            self.sublist_output_files = dict()
        else:
            self.sublist_output_files = sublist_output_files

        self.m_optimal_order = optimal_order

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

        (cluster_assignments, cluster_means) = utils.kmeans_sparse(
            kmeans_data, Wt, self.m_K, norm=1)

        # sort by word counts (decreasing)
        cluster_word_counts = np.zeros(self.m_K, dtype=np.uint)
        for i in xrange(self.m_K):
            cluster_docs = np.where(cluster_assignments == i)[0]
            for j in cluster_docs:
                cluster_word_counts[i] += docs[j].total
        reverse_cluster_order = np.argsort(cluster_word_counts)
        cluster_means[:] = cluster_means[reverse_cluster_order[::-1]]

        logging.debug('Initialization means:')
        for i in xrange(self.m_K):
            w_order = np.argsort(cluster_means[i,:])
            logging.debug('\t%s' % ' '.join(str(batch_to_vocab_word_map[w_order[j]]) for j in xrange(Wt-1, max(-1,Wt-11), -1)))

        self.m_lambda_ss = eff_init_samples * init_noise_weight * nprand.dirichlet(100 * np.ones(self.m_W) / float(self.m_W), self.m_K)
        self.m_lambda_ss[:, batch_to_vocab_word_map] += eff_init_samples * (1 - init_noise_weight) * cluster_means

        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)
        self.m_Elogprobw = utils.log_dirichlet_expectation(self.m_lambda0 + self.m_lambda_ss)

    def process_documents(self, docs, var_converge, update=True,
                          predict_docs=None, save_model=False):
        docs = list(docs)
        num_docs = len(docs)

        if predict_docs is None:
            predict_docs = [None] * num_docs

        # Find the unique words in this mini-batch of documents...
        self.m_num_docs_processed += num_docs

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
            % (num_docs, num_tokens, Wt))

        ss = suff_stats(self.m_K, Wt, num_docs)

        # First row of ElogV is E[log(V)], second row is E[log(1 - V)]
        ElogV = utils.Elog_sbc_stop(self.m_tau)

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
            if self.m_optimal_order:
                order = np.argsort(self.m_lambda_ss_sum)[::-1]
                self.m_lambda_ss = self.m_lambda_ss[order]
                self.m_lambda_ss_sum = self.m_lambda_ss_sum[order]
                self.m_tau_ss = self.m_tau_ss[order]
            self.update_lambda()
            self.update_tau()
            self.m_t += 1

        return (score, count, num_docs)

    def update_lambda(self):
        self.m_Elogprobw = (
            sp.psi(self.m_lambda0 + self.m_lambda_ss)
            - sp.psi(self.m_W*self.m_lambda0 + self.m_lambda_ss_sum[:, np.newaxis])
        )

    def update_nu(self, Elogprobw_doc, doc, Elogpi, phi, nu, log_nu, incorporate_prior=True):
        log_nu[:,:] = np.dot(np.repeat(Elogprobw_doc, doc.counts, axis=1).T, phi.T)
        if incorporate_prior:
            log_nu[:,:] += Elogpi
        for n in xrange(doc.total):
            (log_nu[n,:], log_norm) = utils.log_normalize(log_nu[n,:])
        nu[:,:] = np.exp(log_nu)

    def update_uv(self, nu, uv):
        nu_sums = np.sum(nu, 0)
        uv[0] = 1.0 + nu_sums
        uv[1] = self.m_beta
        uv[1,:self.m_L-1] += np.flipud(np.cumsum(np.flipud(nu_sums[1:])))
        uv[:,self.m_L-1] = [1., 0.]

    def update_tau(self):
        self.m_tau[0] = 1.0 + self.m_tau_ss
        self.m_tau[1] = self.m_alpha
        self.m_tau[1,:self.m_K-1] += np.flipud(np.cumsum(np.flipud(self.m_tau_ss[1:])))
        self.m_tau[:,self.m_K-1] = [1., 0.]

    def update_phi(self, Elogprobw_doc, doc, ElogV, nu, phi, log_phi, incorporate_prior=True):
        log_phi[:,:] = np.dot(np.repeat(Elogprobw_doc, doc.counts, axis=1), nu).T
        if incorporate_prior:
            log_phi[:,:] += ElogV
        for i in xrange(self.m_L):
            (log_phi[i,:], log_norm) = utils.log_normalize(log_phi[i,:])
        phi[:,:] = np.exp(log_phi)

    def z_score(self, ElogV, phi, log_phi):
        return np.sum(phi * (ElogV - log_phi))

    def c_score(self, Elogpi, nu, log_nu):
        return np.sum(nu * (Elogpi - log_nu))

    def w_score(self, doc, nu, phi, Elogprobw_doc):
        return np.sum(nu.T * np.dot(phi, np.repeat(Elogprobw_doc, doc.counts, axis=1)))

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

        Elogprobw_doc = self.m_Elogprobw[:, doc.words]

        logging.debug('Initializing document variational parameters')

        uv = np.zeros((2, self.m_L))
        uv[0] = 1.
        uv[1] = self.m_beta
        uv[1,self.m_L-1] = 0.

        Elogpi = utils.Elog_sbc_stop(uv)

        phi = np.zeros((self.m_L, self.m_K)) / self.m_K
        log_phi = np.log(phi)

        nu = np.ones((num_tokens, self.m_L)) / self.m_L
        log_nu = np.log(nu)

        converge = None
        likelihood = None
        score = None
        old_score = None

        iteration = 0
        # not yet support second level optimization yet, to be done in the
        # future
        while iteration < max_iter and (converge is None or converge < 0.0 or converge > var_converge):
            logging.debug('Updating document variational parameters (iteration: %d)' % iteration)
            # update variational parameters
            self.update_phi(Elogprobw_doc, doc, ElogV, nu, phi, log_phi)
            self.update_nu(Elogprobw_doc, doc, Elogpi, phi, nu, log_nu)
            # TODO why after phi and nu update, not before?
            self.update_uv(nu, uv)
            Elogpi = utils.Elog_sbc_stop(uv)

            # compute score

            score = 0.0

            # E[log p(V | beta)] + H(q(V))
            v_ll = utils.log_sticks_score(uv[:,:self.m_L-1], 1.0, self.m_beta)
            score += v_ll
            logging.debug('Score after V components: %f (+ %f)' % (score, v_ll))

            # E[log p(z | V)] + H(q(z))
            z_ll = self.z_score(ElogV, phi, log_phi)
            score += z_ll
            logging.debug('Score after z components: %f (+ %f)' % (score, z_ll))

            # E[log p(c | U, V)] + H(q(c))
            c_ll = self.c_score(Elogpi, nu, log_nu)
            score += c_ll
            logging.debug('Score after c components: %f (+ %f)' % (score, c_ll))

            # E[log p(W | theta, c, z)]
            w_ll = self.w_score(doc, nu, phi, Elogprobw_doc)
            likelihood = w_ll
            score += w_ll
            logging.debug('Score after W component: %f (+ %f)' % (score, w_ll))

            logging.debug('Score: %f' % score)

            if old_score is not None:
                converge = (score - old_score) / abs(old_score)
                if converge < 0:
                    logging.warning('Score is decreasing')
            old_score = score

            iteration += 1

        ss.m_tau_ss += np.sum(phi, 0)
        for n in xrange(num_tokens):
            ss.m_lambda_ss[:, token_batch_ids[n]] += np.dot(phi.T, nu[n,:])

        if save_model:
            # save sublist stats
            self.save_sublist(
                self.sublist_output_files.get('sublist', None),
                doc, phi)
            self.save_sublist_Elogpi(
                self.sublist_output_files.get('sublist_Elogpi', None),
                doc, uv)
            self.save_sublist_logEpi(
                self.sublist_output_files.get('sublist_logEpi', None),
                doc, uv)
            self.save_sublist_lambda_ss(
                self.sublist_output_files.get('sublist_lambda_ss', None),
                doc, nu)

        if predict_doc is not None:
            likelihood = 0.
            logEVd = utils.logE_sbc_stop(uv)
            # TODO abstract this?
            logEtheta = (
                np.log(self.m_lambda0 + self.m_lambda_ss)
                - np.log(self.m_W*self.m_lambda0 + self.m_lambda_ss_sum[:,np.newaxis])
            )
            for (w, w_count) in zip(predict_doc.words, predict_doc.counts):
                expected_word_prob = 0
                for i in xrange(self.m_L):
                    for j in xrange(self.m_K):
                        expected_word_prob += np.exp(
                            logEVd[i]
                            + log_phi[i,j]
                            + logEtheta[j,w]
                        )
                likelihood += np.log(expected_word_prob) * w_count

        return likelihood

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
        logEpi = utils.logE_sbc_stop(self.m_tau)
        self.save_rows(f, logEpi[:,np.newaxis])

    def save_Elogpi(self, f):
        Elogpi = utils.Elog_sbc_stop(self.m_tau)
        self.save_rows(f, Elogpi[:,np.newaxis])

    def save_pickle(self, f):
        cPickle.dump(self, f, -1)

    def save_sublist_lambda_ss(self, f, doc, nu):
        nu_sums = np.sum(nu, 0)
        self.save_sublist_row(f, doc, nu_sums + self.m_lambda0)

    def save_sublist_logEpi(self, f, doc, uv):
        logEpi = utils.logE_sbc_stop(uv)
        self.save_sublist_row(f, doc, logEpi)

    def save_sublist_Elogpi(self, f, doc, uv):
        Elogpi = utils.Elog_sbc_stop(uv)
        self.save_sublist_row(f, doc, Elogpi)

    def save_sublist(self, f, doc, phi):
        flat_phi = phi.reshape(self.m_L * self.m_K)
        self.save_sublist_row(f, doc, flat_phi)

    def save_rows(self, f, m):
        if f is not None:
            for v in m:
                line = ' '.join([str(x) for x in v])
                f.write('%s\n' % line)

    def save_sublist_row(self, f, doc, v):
        if f is not None:
            line = ' '.join([str(x) for x in v])
            f.write('%s %s\n' % (str(doc.id), line))
