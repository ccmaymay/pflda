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
    def __init__(self, K, J, Wt, Dt, Mt):
        self.m_batch_D = Dt
        self.m_batch_M = Mt
        self.m_tau_ss = np.zeros(K)
        self.m_lambda_ss = np.zeros((K, Wt))
        self.m_omega_ss = np.zeros(Mt)
        self.m_ab_ss = np.zeros((Mt, J))
        self.m_zeta_ss = np.zeros((Mt, J, K))
        self.m_class_lambda_ss_sums = np.zeros((Mt, K))
        self.m_class_lambda_ss_sums_m = np.zeros((Mt, J))


class model(object):
    def __init__(self,
                 K,
                 J,
                 I,
                 M,
                 D,
                 W,
                 lambda0=0.01,
                 omega0=0.1,
                 alpha=1.,
                 beta=1.,
                 gamma=1.,
                 kappa=0.5,
                 iota=1.,
                 scale=1.,
                 rho_bound=0.,
                 optimal_order=False):
        self.m_K = K
        self.m_J = J
        self.m_I = I
        self.m_M = M
        self.m_W = W
        self.m_D = D
        self.m_alpha = alpha
        self.m_beta = beta
        self.m_gamma = gamma

        self.m_classes = [None] * M
        self.m_r_classes = dict()
        self.m_class_lambda_ss_sums = np.zeros((M, self.m_K))
        self.m_class_lambda_ss_sums_m = np.zeros((M, self.m_J))

        self.m_tau = np.zeros((2, self.m_K))
        self.m_tau[0] = 1.0
        self.m_tau[1] = alpha
        self.m_tau[1,self.m_K-1] = 0.
        self.m_tau_ss = np.zeros(self.m_K)
        self.m_Elogpi = utils.Elog_sbc_stop(self.m_tau)

        # Intuition: take 100 to be the expected document length (TODO)
        # so that there are 100D tokens in total.  Then divide that
        # count somewhat evenly (i.i.d. Gamma(1,1) distributed) between
        # each word type and topic.  *Then* subtract lambda0 so that the
        # posterior is composed of these pseudo-counts only (maximum
        # score / no prior).  (why?!  TODO)
        self.m_lambda0 = lambda0
        self.m_lambda_ss = np.random.gamma(
            1.0, 1.0, (self.m_K, W)) * D * 100 / (self.m_K * W) - lambda0
        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)
        self.m_Elogprobw = utils.log_dirichlet_expectation(self.m_lambda0 + self.m_lambda_ss)

        self.m_omega0 = omega0
        # TODO... makes sense?
        self.m_omega_ss = np.random.gamma(1.0, 1.0, (M,)) * D / M - omega0
        self.m_omega_ss_sum = np.sum(self.m_omega_ss)
        self.m_Elogp = utils.log_dirichlet_expectation(self.m_omega0 + self.m_omega_ss)

        self.m_ab = np.zeros((2, self.m_M, self.m_J))
        self.m_ab[0] = 1.0
        self.m_ab[1] = alpha
        self.m_ab[1,:,self.m_J-1] = 0.
        self.m_ab_ss = np.zeros((self.m_M, self.m_J))
        self.m_Elogpi_m = np.zeros((self.m_M, self.m_J))
        for m in xrange(self.m_M):
            self.m_Elogpi_m[m,:] = utils.Elog_sbc_stop(self.m_ab[:,m,:])

        self.m_zeta = np.ones((self.m_M, self.m_J, self.m_K)) / self.m_K
        self.m_log_zeta = np.log(self.m_zeta)
        self.m_zeta_ss = np.zeros((self.m_M, self.m_J, self.m_K))

        self.m_iota = iota
        self.m_kappa = kappa
        self.m_scale = scale
        self.m_rho_bound = rho_bound
        self.m_t = 0
        self.m_num_docs_processed = 0

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
                          predict_docs=None, predict_classes=False):
        docs = list(docs)
        num_docs = len(docs)
        doc_scores = []
        doc_counts = []

        if predict_docs is None:
            predict_docs = [None] * num_docs

        # Find the unique words in this mini-batch of documents...
        self.m_num_docs_processed += num_docs

        classes_to_batch_map = dict()
        batch_to_classes_map = []
        # mapping from word types in this mini-batch to unique
        # consecutive integers
        vocab_to_batch_word_map = dict()
        # list of unique word types, in order of first appearance
        batch_to_vocab_word_map = []

        for doc in docs:
            if doc.attrs['class'] in self.m_r_classes:
                doc.class_idx = self.m_r_classes[doc.attrs['class']]
            elif not update:
                raise ValueError('cannot predict on unseen class')
            else:
                class_idx = len(self.m_r_classes)
                self.m_classes[class_idx] = doc.attrs['class']
                self.m_r_classes[doc.attrs['class']] = class_idx
                doc.class_idx = class_idx

            if doc.class_idx not in classes_to_batch_map:
                batch_to_classes_map.append(doc.class_idx)
                classes_to_batch_map[doc.class_idx] = len(classes_to_batch_map)

            for w in doc.words:
                if w not in vocab_to_batch_word_map:
                    vocab_to_batch_word_map[w] = len(vocab_to_batch_word_map)
                    batch_to_vocab_word_map.append(w)

        if predict_classes:
            batch_to_classes_map = range(self.m_M)
            classes_to_batch_map = dict(zip(range(self.m_M), range(self.m_M)))

        Mt = len(batch_to_classes_map)

        # number of unique word types in this mini-batch
        num_tokens = sum([sum(doc.counts) for doc in docs])
        Wt = len(batch_to_vocab_word_map)

        logging.info('Processing %d docs spanning %d tokens, %d types'
            % (num_docs, num_tokens, Wt))

        ss = suff_stats(self.m_K, self.m_J, Wt, num_docs, Mt)

        # run variational inference on some new docs
        score = 0.0
        count = 0
        confusion = np.zeros((Mt, Mt), dtype=np.uint)

        if predict_classes:
            for doc in docs:
                true_class_idx = doc.class_idx
                best_doc_score = None
                best_class_idx = None
                for m in xrange(self.m_M):
                    doc.class_idx = m
                    doc_score = self.doc_e_step(doc, ss, vocab_to_batch_word_map,
                        batch_to_vocab_word_map, classes_to_batch_map,
                        batch_to_classes_map, var_converge)
                    if best_doc_score is None or doc_score > best_doc_score:
                        best_doc_score = doc_score
                        best_class_idx = m
                confusion[true_class_idx,best_class_idx] += 1
                doc.class_idx = true_class_idx
                score += best_doc_score
                count += doc.total
        else:
            for (doc, predict_doc) in it.izip(docs, predict_docs):
                doc_score = self.doc_e_step(doc, ss, vocab_to_batch_word_map,
                    batch_to_vocab_word_map, classes_to_batch_map,
                    batch_to_classes_map, var_converge, predict_doc=predict_doc)

                doc_scores.append(doc_score)
                score += doc_score
                if predict_doc is None:
                    count += doc.total
                    doc_counts.append(doc.total)
                else:
                    count += predict_doc.total
                    doc_counts.append(predict_doc.total)

        if update:
            self.update_ss_stochastic(ss, batch_to_vocab_word_map,
                                      batch_to_classes_map)
            if self.m_optimal_order:
                order = np.argsort(self.m_lambda_ss_sum)[::-1]
                self.m_lambda_ss = self.m_lambda_ss[order]
                self.m_lambda_ss_sum = self.m_lambda_ss_sum[order]
                self.m_tau_ss = self.m_tau_ss[order]
                self.m_zeta_ss = self.m_zeta_ss[:,:,order]
                for m in xrange(self.m_M):
                    class_order = np.argsort(self.m_class_lambda_ss_sums_m[m])[::-1]
                    self.m_zeta_ss[m] = self.m_zeta_ss[m,class_order,:]
                    self.m_ab_ss[m] = self.m_ab_ss[m,class_order]
            self.update_lambda()
            self.update_tau()
            self.update_omega()
            self.update_ab()
            self.update_zeta()
            self.m_t += 1

        if predict_classes:
            return (score, count, num_docs, doc_scores, doc_counts, confusion)
        else:
            return (score, count, num_docs, doc_scores, doc_counts)

    def update_lambda(self):
        self.m_Elogprobw = (
            sp.psi(self.m_lambda0 + self.m_lambda_ss)
            - sp.psi(self.m_W*self.m_lambda0 + self.m_lambda_ss_sum[:, np.newaxis])
        )

    def update_tau(self):
        self.m_tau[0] = 1.0 + self.m_tau_ss
        self.m_tau[1] = self.m_alpha
        self.m_tau[1,:self.m_K-1] += np.flipud(np.cumsum(np.flipud(self.m_tau_ss[1:])))
        self.m_tau[:,self.m_K-1] = [1., 0.]
        self.m_Elogpi = utils.Elog_sbc_stop(self.m_tau)

    def update_omega(self):
        self.m_Elogp = (
            sp.psi(self.m_omega0 + self.m_omega_ss)
            - sp.psi(self.m_M*self.m_omega0 + self.m_omega_ss_sum)
        )

    def update_ab(self):
        self.m_ab[0,:,:] = 1.0 + self.m_ab_ss
        self.m_ab[1,:,:] = self.m_beta
        for m in xrange(self.m_M):
            self.m_ab[1,m,:self.m_J-1] += np.flipud(np.cumsum(np.flipud(self.m_ab_ss[m,1:])))
            self.m_ab[:,m,self.m_J-1] = [1., 0.]
            self.m_Elogpi_m[m,:] = utils.Elog_sbc_stop(self.m_ab[:,m,:])

    def update_zeta(self):
        self.m_log_zeta = self.m_zeta_ss + self.m_Elogpi
        for m in xrange(self.m_M):
            for j in xrange(self.m_J):
                (self.m_log_zeta[m,j,:], log_norm) = utils.log_normalize(self.m_log_zeta[m,j,:])
        self.m_zeta = np.exp(self.m_log_zeta)

    def update_nu(self, Elogprobw_doc, zeta_doc, doc, Elogpi_d, phi, nu, log_nu, incorporate_prior=True):
        log_nu[:,:] = np.dot(phi, np.dot(zeta_doc, np.repeat(Elogprobw_doc, doc.counts, axis=1))).T
        if incorporate_prior:
            log_nu[:,:] += Elogpi_d
        for n in xrange(doc.total):
            (log_nu[n,:], log_norm) = utils.log_normalize(log_nu[n,:])
        nu[:,:] = np.exp(log_nu)

    def update_uv(self, nu, uv):
        nu_sums = np.sum(nu, 0)
        uv[0] = 1.0 + nu_sums
        uv[1] = self.m_gamma
        uv[1,:self.m_I-1] += np.flipud(np.cumsum(np.flipud(nu_sums[1:])))
        uv[:,self.m_I-1] = [1., 0.]

    def update_phi(self, Elogprobw_doc, zeta_doc, doc, Elogpi_m_doc, nu, phi, log_phi, incorporate_prior=True):
        log_phi[:,:] = np.dot(np.dot(zeta_doc, np.repeat(Elogprobw_doc, doc.counts, axis=1)), nu).T
        if incorporate_prior:
            log_phi[:,:] += Elogpi_m_doc
        for i in xrange(self.m_I):
            (log_phi[i,:], log_norm) = utils.log_normalize(log_phi[i,:])
        phi[:,:] = np.exp(log_phi)

    def z_score(self, Elogpi_m_doc, phi, log_phi):
        return np.sum(phi * (Elogpi_m_doc - log_phi))

    def c_score(self, Elogpi_d, nu, log_nu):
        return np.sum(nu * (Elogpi_d - log_nu))

    def x_score(self, Elogp_doc):
        return Elogp_doc

    def w_score(self, zeta_doc, doc, nu, phi, Elogprobw_doc):
        return np.sum(nu.T * np.dot(phi, np.dot(zeta_doc, np.repeat(Elogprobw_doc, doc.counts, axis=1))))

    def doc_e_step(self, doc, ss, vocab_to_batch_word_map,
                   batch_to_vocab_word_map, classes_to_batch_map,
                   batch_to_classes_map, var_converge, max_iter=100,
                   predict_doc=None):
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

        uv = np.zeros((2, self.m_I))
        uv[0] = 1.
        uv[1] = self.m_gamma
        uv[1,self.m_I-1] = 0.

        Elogpi_m_doc = self.m_Elogpi_m[doc.class_idx]
        Elogpi_d = utils.Elog_sbc_stop(uv)
        zeta_doc = self.m_zeta[doc.class_idx]
        log_zeta_doc = self.m_log_zeta[doc.class_idx]

        phi = np.zeros((self.m_I, self.m_J)) / self.m_J
        log_phi = np.log(phi)

        nu = np.ones((num_tokens, self.m_I)) / self.m_I
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
            self.update_phi(Elogprobw_doc, zeta_doc, doc, Elogpi_m_doc, nu, phi, log_phi)
            self.update_nu(Elogprobw_doc, zeta_doc, doc, Elogpi_d, phi, nu, log_nu)
            # TODO why after phi and nu update, not before?
            self.update_uv(nu, uv)
            Elogpi_d = utils.Elog_sbc_stop(uv)

            # compute score

            score = 0.0

            # E[log p(x | p)
            x_ll = self.x_score(self.m_Elogp[doc.class_idx])
            score += x_ll
            logging.debug('Score after x component: %f (+ %f)' % (score, x_ll))

            # E[log p(Vd | gamma)] + H(q(Vd))
            v_ll = utils.log_sticks_score(uv[:,:self.m_I-1], 1.0, self.m_gamma)
            score += v_ll
            logging.debug('Score after Vd components: %f (+ %f)' % (score, v_ll))

            # E[log p(z | Vm)] + H(q(z))
            z_ll = self.z_score(Elogpi_m_doc, phi, log_phi)
            score += z_ll
            logging.debug('Score after z components: %f (+ %f)' % (score, z_ll))

            # E[log p(c | Vd)] + H(q(c))
            c_ll = self.c_score(Elogpi_d, nu, log_nu)
            score += c_ll
            logging.debug('Score after c components: %f (+ %f)' % (score, c_ll))

            # E[log p(W | c, z, y, theta)]
            w_ll = self.w_score(zeta_doc, doc, nu, phi, Elogprobw_doc)
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

        ss.m_tau_ss += np.sum(zeta_doc, 0)
        for n in xrange(num_tokens):
            ss.m_lambda_ss[:, token_batch_ids[n]] += np.dot(np.dot(nu[n,:], phi), zeta_doc)
        ss.m_omega_ss[classes_to_batch_map[doc.class_idx]] += 1
        ss.m_ab_ss[classes_to_batch_map[doc.class_idx],:] += np.sum(phi, 0)
        ss.m_zeta_ss[classes_to_batch_map[doc.class_idx],:,:] += np.dot(np.repeat(Elogprobw_doc, doc.counts, axis=1), np.dot(nu, phi)).T
        ss.m_class_lambda_ss_sums[classes_to_batch_map[doc.class_idx]] += np.dot(np.dot(np.sum(nu, 0), phi), zeta_doc)
        ss.m_class_lambda_ss_sums_m[classes_to_batch_map[doc.class_idx]] += np.dot(np.sum(nu, 0), phi)

        if predict_doc is not None:
            likelihood = 0.
            logEpi_d = utils.logE_sbc_stop(uv)
            # TODO abstract this?
            logEtheta = (
                np.log(self.m_lambda0 + self.m_lambda_ss)
                - np.log(self.m_W*self.m_lambda0 + self.m_lambda_ss_sum[:,np.newaxis])
            )
            for (w, w_count) in zip(predict_doc.words, predict_doc.counts):
                expected_word_prob = 0
                for i in xrange(self.m_I):
                    for j in xrange(self.m_J):
                        for k in xrange(self.m_K):
                            expected_word_prob += np.exp(
                                logEpi_d[i]
                                + log_phi[i,j]
                                + log_zeta_doc[j,k]
                                + logEtheta[k,w]
                            )
                likelihood += np.log(expected_word_prob) * w_count

        return likelihood

    def update_ss_stochastic(self, ss, batch_to_vocab_word_map,
                             batch_to_classes_map):
        # rho will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        # Add one because self.m_t is zero-based.
        rho = self.m_scale * pow(self.m_iota + self.m_t + 1, -self.m_kappa)
        if rho < self.m_rho_bound:
            rho = self.m_rho_bound

        # Update lambda based on documents.
        self.m_lambda_ss *= (1 - rho)
        self.m_lambda_ss[:, batch_to_vocab_word_map] += (
            rho * ss.m_lambda_ss * self.m_D / ss.m_batch_D
        )
        self.m_lambda_ss_sum = np.sum(self.m_lambda_ss, axis=1)

        self.m_tau_ss = (
            (1.0 - rho) * self.m_tau_ss
            + rho * ss.m_tau_ss * self.m_D / ss.m_batch_D
        )

        self.m_omega_ss *= (1 - rho)
        self.m_omega_ss[batch_to_classes_map] += (
            rho * ss.m_omega_ss * self.m_D / ss.m_batch_D
        )
        self.m_omega_ss_sum = np.sum(self.m_omega_ss)

        self.m_ab_ss *= (1 - rho)
        self.m_ab_ss[batch_to_classes_map,:] += (
            rho * ss.m_ab_ss * self.m_D / ss.m_batch_D
        )

        self.m_zeta_ss *= (1 - rho)
        self.m_zeta_ss[batch_to_classes_map,:,:] += (
            rho * ss.m_zeta_ss * self.m_D / ss.m_batch_D
        )

        self.m_class_lambda_ss_sums *= (1 - rho)
        self.m_class_lambda_ss_sums[batch_to_classes_map,:] += (
            rho * ss.m_class_lambda_ss_sums * self.m_D / ss.m_batch_D
        )

        self.m_class_lambda_ss_sums_m *= (1 - rho)
        self.m_class_lambda_ss_sums_m[batch_to_classes_map,:] += (
            rho * ss.m_class_lambda_ss_sums_m * self.m_D / ss.m_batch_D
        )

    def save(self, output_files):
        self.save_lambda_ss(output_files.get('lambda_ss', None))
        self.save_logEtheta(output_files.get('logEtheta', None))
        self.save_Elogtheta(output_files.get('Elogtheta', None))
        self.save_logEpi(output_files.get('logEpi', None))
        self.save_Elogpi(output_files.get('Elogpi', None))
        self.save_pickle(output_files.get('pickle', None))

        for class_idx in xrange(self.m_M):
            self.save_sublist(
                output_files.get('sublist', None), class_idx)
            self.save_sublist_Elogpi(
                output_files.get('sublist_Elogpi', None), class_idx)
            self.save_sublist_logEpi(
                output_files.get('sublist_logEpi', None), class_idx)
            self.save_sublist_lambda_ss(
                output_files.get('sublist_lambda_ss', None), class_idx)

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

    def save_sublist_lambda_ss(self, f, class_idx):
        self.save_sublist_row(f, class_idx, self.m_class_lambda_ss_sums_m[class_idx] + self.m_lambda0)

    def save_sublist_logEpi(self, f, class_idx):
        logEpi_m = utils.logE_sbc_stop(self.m_ab[:,class_idx,:])
        self.save_sublist_row(f, class_idx, logEpi_m)

    def save_sublist_Elogpi(self, f, class_idx):
        Elogpi_m = utils.Elog_sbc_stop(self.m_ab[:,class_idx,:])
        self.save_sublist_row(f, class_idx, Elogpi_m)

    def save_sublist(self, f, class_idx):
        flat_zeta = self.m_zeta[class_idx].reshape((self.m_J * self.m_K,))
        self.save_sublist_row(f, class_idx, flat_zeta)

    def save_rows(self, f, m):
        if f is not None:
            for v in m:
                line = ' '.join([str(x) for x in v])
                f.write('%s\n' % line)

    def save_sublist_row(self, f, class_idx, v):
        if f is not None:
            line = ' '.join([str(x) for x in v])
            f.write(u'%s %s\n' % (self.m_classes[class_idx], line))
