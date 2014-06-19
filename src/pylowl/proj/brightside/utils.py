import random
import numpy as np
import numpy.random as nprand
import numpy.linalg as la
import scipy.special as sp
import itertools as it
import os


def write_concrete(docs, output_dir):
    from thrift.transport import TTransport
    from thrift.protocol import TBinaryProtocol
    from concrete.communication.ttypes import Communication
    from concrete.structure.ttypes import (
        SectionSegmentation, Section,
        SentenceSegmentation, Sentence,
        Tokenization, Token
    )

    def make_comm(tokens):
        comm = Communication()
        comm.text = ' '.join(tokens)
        sectionSegmentation = SectionSegmentation()
        section = Section()
        sentenceSegmentation = SentenceSegmentation()
        sentence = Sentence()
        tokenization = Tokenization()
        tokenization.tokenList = [Token(text=t) for t in tokens]
        sentence.tokenizationList = [tokenization]
        sentenceSegmentation.sentenceList = [sentence]
        section.sentenceSegmentation = [sentenceSegmentation]
        sectionSegmentation.sectionList = [section]
        comm.sectionSegmentations = [sectionSegmentation]
        return comm

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    i = 0
    output_path = os.path.join(output_dir, '%d.dat' % i)
    for doc in docs:
        comm = make_comm(doc)
        while os.path.exists(output_path):
            i += 1
            output_path = os.path.join(output_dir, '%d.dat' % i)
        with open(output_path, 'wb') as f:
            transport = TTransport.TFileObjectTransport(f)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            comm.write(protocol)
        i += 1
        output_path = os.path.join(output_dir, '%d.dat' % i)


def load_concrete(loc, section_segmentation_idx=0, sentence_segmentation_idx=0,
                  tokenization_list_idx=0):
    from thrift.transport import TTransport
    from thrift.protocol import TBinaryProtocol
    from concrete.communication.ttypes import Communication

    def parse_comm(comm):
        tokens = []
        if comm.sectionSegmentations is not None:
            section_segmentation = comm.sectionSegmentations[section_segmentation_idx]
            if section_segmentation.sectionList is not None:
                for section in section_segmentation.sectionList:
                    if section.sentenceSegmentation is not None:
                        sentence_segmentation = section.sentenceSegmentation[sentence_segmentation_idx]
                        if sentence_segmentation.sentenceList is not None:
                            for sentence in sentence_segmentation.sentenceList:
                                if sentence.tokenizationList is not None:
                                    tokenization = sentence.tokenizationList[tokenization_list_idx]
                                    if tokenization.tokenList is not None:
                                        for token in tokenization.tokenList:
                                            tokens.append(token.text)
        return tokens

    for input_path in path_list(loc):
        with open(input_path, 'rb') as f:
            transportIn = TTransport.TFileObjectTransport(f)
            protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
            comm = Communication()
            comm.read(protocolIn)
            tokens = parse_comm(comm)
            yield (input_path, tokens)


def path_list(loc):
    if isinstance(loc, str):
        return [loc]
    else:
        return loc


def take(g, n):
    return (x for (i, x) in it.izip(xrange(n), g))


def _pair_first_to_int(p):
    return (int(p[0]), p[1])


def load_vocab(filename):
    with open(filename) as f:
        vocab = dict(_pair_first_to_int(line.strip().split()) for line in f)
    return vocab


def vector_norm(m, axis=0, ord=None):
    '''
    Return vector norm across specified axis of matrix (axis 0 by
    default), using the specified order.
    '''
    new_shape = m.shape[:axis] + m.shape[(axis+1):]
    norm = np.zeros(new_shape)
    for idx in np.ndindex(*new_shape):
        m_idx = idx[:axis] + (range(m.shape[axis]),) + idx[axis:]
        norm[idx] = la.norm(m.__getitem__(m_idx), ord=ord)
    return norm

    
def log_sticks_likelihood(ab, a_prior, b_prior, ids):
    '''
    Return E[log p(X | a_prior, b_prior)] + H(q(X))
    where X is a random vector whose components are Beta-distributed
    with parameters a_prior and b_prior, the variational distribution of
    X is that each component of X is an independent Beta random
    variable whose parameters are given by the corresponding column of
    ab, and the expectation E is taken with respect to the variational
    distribution.  ids is an array of indices used to subselect
    components of X (we ignore components whose indices are not in ids).
    '''
    sum_ab = np.sum(ab[:,ids], 0)
    diff_psi_ab = sp.psi(ab[:,ids]) - sp.psi(sum_ab)
    ab_entr_factors = ab[:,ids]
    ab_prob_factors = np.zeros(2)
    ab_prob_factors[0] = a_prior
    ab_prob_factors[1] = b_prior
    ab_entr_log_beta = (
        np.sum(sp.gammaln(ab[:, ids]), 0)
        - sp.gammaln(sum_ab)
    )
    ab_prob_log_beta = (
        sp.gammaln(a_prior) + sp.gammaln(b_prior)
        - sp.gammaln(a_prior + b_prior)
    )
    return np.sum(
        np.sum((ab_prob_factors[:, np.newaxis] - ab_entr_factors)
               * diff_psi_ab, 0)
        + ab_entr_log_beta - ab_prob_log_beta
    )


def log_normalize(v):
    '''
    Return pair: normalization of v and the normalizer.  (v is assumed
    to be in log space; the normalization r satisfies sum(exp(r)) = 1.
    The normalizer is also represented in log space.)  v can also be a
    matrix, in which case the normalization is performed column-wise
    (each row r satisfies sum(exp(r)) = 1) and the normalizer is a
    vector containing the normalizer for each row.
    '''

    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v) + 1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1] + 1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:, np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:, np.newaxis]

    return (v, log_norm)


def log_sum(log_a, log_b):
    '''
    Return log(a+b), given log(a) and log(b).  Perform the computation
    in log space to prevent underflow.
    '''
    v = 0.0
    if (log_a < log_b):
        v = log_b + np.log(1 + np.exp(log_a - log_b))
    else:
        v = log_a + np.log(1 + np.exp(log_b - log_a))
    return v


def log_dirichlet_expectation(conc):
    '''
    Compute expectation of the log of a Dirichlet-distributed r.v. with
    parameter vector conc (or an array of Dirichlet r.v.s with
    parameter vectors given by the rows of conc).
    '''

    if (len(conc.shape) == 1):
        return sp.psi(conc) - sp.psi(np.sum(conc))
    return sp.psi(conc) - sp.psi(np.sum(conc, 1))[:, np.newaxis]


def dirichlet_log_expectation(conc):
    '''
    Compute log of the expectation of a Dirichlet-distributed r.v. with
    parameter vector conc (or an array of Dirichlet r.v.s with
    parameter vectors given by the rows of conc).
    '''

    if (len(conc.shape) == 1):
        return np.log(conc) - np.log(np.sum(conc))
    return np.log(conc) - np.log(np.sum(conc, 1))[:, np.newaxis]


def log_beta_expectation(ab, ids):
    '''
    Return E[log X]
    where X is a random vector whose components are Beta-distributed and
    ids is an array of indices used to subselect components of X (we
    ignore components whose indices are not in ids).
    '''
    sum_ab = np.sum(ab[:,ids], 0)
    return sp.psi(ab[:,ids]) - sp.psi(sum_ab)


def beta_log_expectation(ab, ids):
    '''
    Return log E[X]
    where X is a random vector whose components are Beta-distributed and
    ids is an array of indices used to subselect components of X (we
    ignore components whose indices are not in ids).
    '''
    sum_ab = np.sum(ab[:,ids], 0)
    return np.log(ab[:,ids]) - np.log(sum_ab)


def node_ancestors(node):
    r'''
    Return generator over this node's ancestors, starting from the
    parent and ascending upward.
    '''
    return (node[:-k] for k in xrange(1, len(node)))


def node_left_siblings(node):
    r'''
    Return generator over this node's elder siblings (children of the
    same parent, left of this node in the planar embedding).  Generator
    starts at the sibling to the immediate left and proceeds leftward.
    '''
    return (node[:-1] + (j,) for j in xrange(node[-1] - 1, -1, -1))


def tree_iter(trunc):
    r'''
    Return generator over all nodes in tree, ordered first by level,
    and then left-to-right within each level.
    '''
    for level in range(len(trunc)):
        nodes_by_level = (range(t) for t in trunc[:level+1])
        for node in it.product(*nodes_by_level):
            yield node


def tree_index_b(trunc):
    r'''
    Return vector containing total number of nodes per level in
    (full) truncated tree.  Assume trunc[0] = 1 (the root).
    '''
    return np.cumprod(trunc, dtype=np.uint)


def tree_index_m(trunc):
    r'''
    Return lower-triangular matrix in which the jth element of
    row i, for j <= i, is the number of descendents a level-j node
    has in level i (of the tree).  In particular, the diagonal is the
    one vector, everything to the right of the diagonal is zero,
    and the (i,j) element (for j <= i) is given by
        \prod_{k=j+1}^{i}{ trunc[k] }.
    Assume trunc[0] = 1 (the root).
    '''
    u = list(trunc[1:]) + [1]
    m = np.zeros((len(trunc),len(trunc)), dtype=np.uint)
    for i in xrange(len(trunc)):
        m[i,:i+1] = np.flipud(np.cumprod(np.flipud(u[:i] + [1])))
    return m


def tree_index(x, m, b):
    '''
    Return the index into a flat array representing the tree
    corresponding to the per-level offsets given by x.  The
    parameters m and b correspond to the outputs of
    tree_index_m and tree_index_b, respectively.
    '''
    return int(np.sum(b[:len(x)-1]) + np.dot(m[len(x)-1,:len(x)], x))


def subtree_node_candidates(trunc, subtree):
    '''
    TODO
    '''

    global_nodes_in_subtree = set(subtree.values())
    for node in tree_iter(trunc):
        if node not in subtree and node[:-1] in subtree:
            # node not in subtree but has parent in subtree
            if node[-1] == 0 or (node[:-1] + (node[-1] - 1,)) in subtree:
                # node is first child or has left sibling in subtree
                global_p = subtree[node[:-1]]
                for j in xrange(trunc[len(global_p)]):
                    global_node = global_p + (j,)
                    if global_node not in global_nodes_in_subtree:
                        yield (node, global_node)


def kmeans(data, k, norm=None):
    '''
    Infer hard k-means clustering of data using Lloyd's algorithm.
    data is a N x D matrix containing N objects, each of which is
    represented by D features.  Distances are computed with respect
    to specified norm (2-norm by default).  Return pair containing
      0. cluster assignments (N-vector of indices between 0 and k-1)
      1. cluster means (k x D matrix of cluster means).
    '''

    # TODO abstract some of this

    if data.shape[0] <= k:
        cluster_means = np.zeros((k, data.shape[1]))
        cluster_means[:data.shape[0], :] = data
        return (np.arange(data.shape[0]), cluster_means)

    cluster_assignments = np.zeros(data.shape[0], dtype=np.uint)
    t = 0
    for i in xrange(data.shape[0]):
        cluster_assignments[i] = t
        t = (t + 1) if (t + 1 < k) else 0
    nprand.shuffle(cluster_assignments)
    cluster_assignments_binary = np.zeros((k, data.shape[0]), dtype=np.bool)
    cluster_assignments_binary[cluster_assignments, np.arange(data.shape[0])] = True
    
    cluster_sizes = np.sum(cluster_assignments_binary, 1)
    cluster_means = np.array(np.dot(cluster_assignments_binary, data), dtype=np.double)
    cluster_means[cluster_sizes > 0,:] /= cluster_sizes[cluster_sizes > 0][:,np.newaxis]
    cluster_means[cluster_sizes == 0,:] = np.zeros(data.shape[1])

    old_cluster_assignments = None

    while old_cluster_assignments is None or (old_cluster_assignments != cluster_assignments).any():
        cluster_diffs = (
            cluster_means[:, :, np.newaxis]
            - data.T[np.newaxis, :, :].repeat(k, 0)
        )
        cluster_distances = vector_norm(cluster_diffs, ord=norm, axis=1)

        old_cluster_assignments = cluster_assignments

        cluster_assignments = np.argmin(cluster_distances, 0)
        cluster_assignments_binary[:,:] = False
        cluster_assignments_binary[cluster_assignments, np.arange(data.shape[0])] = True

        cluster_sizes = np.sum(cluster_assignments_binary, 1)
        cluster_means = np.array(np.dot(cluster_assignments_binary, data), dtype=np.double)
        cluster_means[cluster_sizes > 0,:] /= cluster_sizes[cluster_sizes > 0][:,np.newaxis]
        cluster_means[cluster_sizes == 0,:] = np.zeros(data.shape[1])

    return (cluster_assignments, cluster_means)


def kmeans_sparse(data, num_features, k, norm=None):
    '''
    Infer hard k-means clustering of data using Lloyd's algorithm.
    data is a list of N tuples, each of which represents an object
    represented by D features (sparsely).  In particular, the first
    element of a tuple is a list of feature ids and the second
    element is a corresponding vector of feature values.  num_features
    is the total number of features occurring in the data; feature ids
    are assumed to be integers between 0 (inclusive) and num_features
    (exclusive).  Distances are computed with respect to specified norm
    (2-norm by default).  Return pair containing
      0. cluster assignments (N-vector of indices between 0 and k-1)
      1. cluster means (k x D matrix of cluster means).
    '''

    num_samples = len(data)

    if num_samples <= k:
        cluster_means = np.zeros((k, num_features))
        for (i, x) in enumerate(data):
            cluster_means[i,x[0]] = x[1]
        return (np.arange(num_samples), cluster_means)

    cluster_assignments = np.zeros(num_samples, dtype=np.uint)
    cluster_sizes = np.zeros(k, dtype=np.uint)
    cluster_means = np.zeros((k, num_features))
    t = 0
    for (i, x) in enumerate(data):
        cluster_assignments[i] = t
        t = (t + 1) if (t + 1 < k) else 0
    nprand.shuffle(cluster_assignments)

    for (i, x) in enumerate(data):
        t = cluster_assignments[i]
        cluster_sizes[t] += 1
        cluster_means[t,x[0]] += x[1]
    cluster_means[cluster_sizes > 0,:] /= cluster_sizes[cluster_sizes > 0][:,np.newaxis]

    updated_cluster_assignment = True
    full_x = np.zeros(num_features)

    while updated_cluster_assignment:
        updated_cluster_assignment = False

        cluster_sizes[:] = 0
        for (i, x) in enumerate(data):
            full_x[:] = 0
            full_x[x[0]] = x[1]
            t = np.argmin(vector_norm(cluster_means - full_x, ord=norm, axis=1))
            if t != cluster_assignments[i]:
                updated_cluster_assignment = True
            cluster_assignments[i] = t
            cluster_sizes[t] += 1

        cluster_means[:,:] = 0
        for (i, x) in enumerate(data):
            t = cluster_assignments[i]
            cluster_means[t,x[0]] += x[1]
        cluster_means[cluster_sizes > 0,:] /= cluster_sizes[cluster_sizes > 0][:,np.newaxis]

    return (cluster_assignments, cluster_means)
