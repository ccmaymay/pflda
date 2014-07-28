import random
import numpy as np
import numpy.random as nprand
import numpy.linalg as la
import scipy.special as sp
import itertools as it
import os
import tempfile


OPTIONS_KV_DELIM = ': '


def parent_package_name(package_name):
    if '.' in package_name:
        return package_name[:package_name.rfind('.')]
    else:
        return None


def make_output_dir(parent_dir):
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    output_dir = tempfile.mkdtemp(prefix='', suffix='', dir=parent_dir)
    umask = os.umask(0o022) # whatever, python
    os.umask(umask) # set umask back
    os.chmod(output_dir, 0o0755 & ~umask)
    return output_dir


def reservoir_insert(reservoir, n, item):
    if n < len(reservoir):
        reservoir[n] = item
        return n
    else:
        if random.random() < len(reservoir)/float(n + 1):
            i = random.randint(0, len(reservoir) - 1)
            reservoir[i] = item
            return i
        return None


def load_options(path):
    options = dict()
    with open(path) as f:
        for line in f:
            line = line.strip()
            i = line.find(OPTIONS_KV_DELIM)
            k = line[:i]
            v = line[i+len(OPTIONS_KV_DELIM):]
            options[k] = v
    return options


def get_path_suffix(path, stem):
    path_stem = os.path.normpath(path)
    path_suffix = None
    while os.path.normpath(os.path.abspath(path_stem)) != os.path.normpath(os.path.abspath(stem)):
        if not path_stem:
            raise Exception('"%s" is not an ancestor of "%s"'
                            % (stem, path))

        (path_stem, basename) = os.path.split(path_stem)
        if path_suffix is None:
            path_suffix = basename
        else:
            path_suffix = os.path.join(basename, path_suffix)
    if path_suffix is None:
        return os.path.curdir
    else:
        return path_suffix


def mkdirp_parent(path):
    mkdirp(os.path.dirname(path))


def mkdirp(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def path_is_not_special(path):
    return not os.path.basename(path).startswith('.')


def path_is_concrete(path):
    return path_is_not_special(path) and path.endswith('.concrete')


def nested_file_paths(root_dir, path_filter=None):
    if path_filter is None:
        path_filter = path_is_not_special
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            p = os.path.join(dirpath, filename)
            if path_filter(p):
                paths.append(p)
    return paths


def nested_input_output_file_paths(input_root_dir, output_root_dir,
                                   path_filter=None):
    input_paths = nested_file_paths(input_root_dir, path_filter)
    output_paths = []
    for input_path in input_paths:
        path_suffix = get_path_suffix(input_path, input_root_dir)
        output_path = os.path.join(output_root_dir, path_suffix)
        mkdirp_parent(output_path)
        output_paths.append(output_path)
    return zip(input_paths, output_paths)


def take(g, n):
    return (x for (i, x) in it.izip(xrange(n), g))


def Elog_sbc_stop(ab):
    '''
    Return expected log stopping probabilities of stick-breaking
    construction with relative weights ab.
    '''
    ElogX = log_beta_expectation(ab)
    n = ab.shape[1]
    Elog_sbc_stop = np.zeros(n)
    Elog_sbc_stop[:n-1] = ElogX[0,:n-1]
    Elog_sbc_stop[1:] += np.cumsum(ElogX[1,:n-1])
    return Elog_sbc_stop


def logE_sbc_stop(ab):
    '''
    Return log expected stopping probabilities of stick-breaking
    construction with relative weights ab.
    '''
    logEX = beta_log_expectation(ab)
    n = ab.shape[1]
    logE_sbc_stop = np.zeros(n)
    logE_sbc_stop[:n-1] = logEX[0,:n-1]
    logE_sbc_stop[1:] += np.cumsum(logEX[1,:n-1])
    return logE_sbc_stop


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

    
def log_sticks_likelihood(ab, a_prior, b_prior):
    '''
    Return E[log p(X | a_prior, b_prior)] + H(q(X))
    where X is a random vector whose components are Beta-distributed
    with parameters a_prior and b_prior, the variational distribution of
    X is that each component of X is an independent Beta random
    variable whose parameters are given by the corresponding column of
    ab, and the expectation E is taken with respect to the variational
    distribution.
    '''
    sum_ab = np.sum(ab, 0)
    diff_psi_ab = sp.psi(ab) - sp.psi(sum_ab)
    ab_entr_factors = ab
    ab_prob_factors = np.zeros(2)
    ab_prob_factors[0] = a_prior
    ab_prob_factors[1] = b_prior
    ab_entr_log_beta = (
        np.sum(sp.gammaln(ab), 0)
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
    The normalizer is also represented in log space.)
    '''

    log_max = 100.0
    max_val = np.max(v)
    log_shift = log_max - np.log(sum(v.shape) + 1.0) - max_val
    tot = np.sum(np.exp(v + log_shift))
    log_norm = np.log(tot) - log_shift
    v = v - log_norm
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


def log_beta_expectation(ab):
    '''
    Return E[log X]
    where X is a random vector whose components are Beta-distributed
    '''
    sum_ab = np.sum(ab, 0)
    return sp.psi(ab) - sp.psi(sum_ab)


def beta_log_expectation(ab):
    '''
    Return log E[X]
    where X is a random vector whose components are Beta-distributed
    '''
    sum_ab = np.sum(ab, 0)
    return np.log(ab) - np.log(sum_ab)


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


def kmeans(data, k, norm=None, max_iters=20):
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

    iteration = 0
    while iteration < max_iters and (old_cluster_assignments is None or (old_cluster_assignments != cluster_assignments).any()):
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

        iteration += 1

    return (cluster_assignments, cluster_means)


def kmeans_sparse(data, num_features, k, norm=None, max_iters=20):
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

    iteration = 0
    while iteration < max_iters and updated_cluster_assignment:
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

        iteration += 1

    return (cluster_assignments, cluster_means)
