import numpy as np
import numpy.linalg as la

from utils import *
from numpy.testing import *


def beta_log_expectation_trivial_unif_test():
    m = beta_log_expectation(np.array([[1.], [1.]]))
    assert_array_almost_equal(m,
        np.log(np.array([[0.5], [0.5]])))

def beta_log_expectation_non_unif_test():
    m = beta_log_expectation(np.array([[42., 1., 1., 3.], [42., 7., 1., 9.]]))
    assert_array_almost_equal(m,
        np.log(np.array([[0.5, 0.125, 0.5, 0.25], [0.5, 0.875, 0.5, 0.75]])))

def vector_norm_3d_0_ord_1_test():
    m = np.ones((4,3,5))
    assert_almost_equal(vector_norm(m, axis=0, ord=1), 4*np.ones((3,5)))

def vector_norm_3d_1_ord_1_test():
    m = np.ones((4,3,5))
    assert_almost_equal(vector_norm(m, axis=1, ord=1), 3*np.ones((4,5)))

def vector_norm_3d_2_ord_1_test():
    m = np.ones((4,3,5))
    assert_almost_equal(vector_norm(m, axis=2, ord=1), 5*np.ones((4,3)))

def vector_norm_3d_0_ord_2_test():
    m = np.ones((4,3,5))
    assert_almost_equal(vector_norm(m, axis=0, ord=2), 2*np.ones((3,5)))

def log_sticks_likelihood_1_test():
    actual = log_sticks_likelihood(
        np.array([[1.], [1.]]),
        1., 1.
    )
    assert_almost_equal(actual, 0.0)

def log_sticks_likelihood_2_test():
    actual = log_sticks_likelihood(
        np.array([[42.], [7.]]),
        42., 7.
    )
    assert_almost_equal(actual, 0.0)

def log_sticks_likelihood_3_test():
    actual = log_sticks_likelihood(
        np.array([[42., 42., 42., 42.], [7., 7., 7., 7.]]),
        42., 7.
    )
    assert_almost_equal(actual, 0.0)

def log_sticks_likelihood_4_test():
    actual = log_sticks_likelihood(
        np.array([[42., 47., 42., 42.], [7., 7., 7., 8.]]),
        42., 7.
    )
    assert_almost_equal(actual, -0.097280122)


def log_normalize_1_test():
    (v, log_norm) = log_normalize(np.array([0]))
    assert_almost_equal(log_norm, 0.0)
    assert_array_almost_equal(v, [0])

def log_normalize_2_test():
    (v, log_norm) = log_normalize(np.zeros(3))
    assert_almost_equal(log_norm, 1.098612289)
    assert_array_almost_equal(v, [-1.098612289, -1.098612289, -1.098612289])

def log_normalize_3_test():
    (v, log_norm) = log_normalize(np.array([0, -np.inf, -np.inf]))
    assert_almost_equal(log_norm, 0.0)
    assert_array_almost_equal(v, [0, -np.inf, -np.inf])

def log_normalize_4_test():
    (v, log_norm) = log_normalize(np.array([0, -np.inf, 1]))
    assert_almost_equal(log_norm, 1.31326169)
    assert_array_almost_equal(v, [-1.31326169, -np.inf, -0.31326169])

def log_normalize_5_test():
    (v, log_norm) = log_normalize(np.array([-5, 5, -2]))
    assert_almost_equal(log_norm, 5.000956823993)
    assert_array_almost_equal(v, [-10.000956823993, -0.000956823993, -7.000956823993])

def log_normalize_6_test():
    (m, log_norm) = log_normalize(np.zeros((3,3)))
    assert_array_almost_equal(log_norm, [1.09861229, 1.09861229, 1.09861229])
    assert_array_almost_equal(m, [[-1.09861229, -1.09861229, -1.09861229], [-1.09861229, -1.09861229, -1.09861229], [-1.09861229, -1.09861229, -1.09861229]])

def log_normalize_7_test():
    (m, log_norm) = log_normalize(np.array([[-5, 5, -2], [0, 0, 0], [0, -np.inf, 1]]))
    assert_array_almost_equal(log_norm, [5.00095682, 1.09861229, 1.31326169])
    assert_array_almost_equal(m, [[-10.0009568, -0.000956823993, -7.000956823993], [-1.09861229, -1.09861229, -1.09861229], [-1.31326169, -np.inf, -0.313261688]])


def node_ancestors_1_test():
    assert list(node_ancestors((0,))) == []

def node_ancestors_2_test():
    assert list(node_ancestors((0, 4))) == [(0,)]

def node_ancestors_3_test():
    assert list(node_ancestors((0, 4, 1, 7))) == [(0, 4, 1), (0, 4), (0,)]


def node_left_siblings_1_test():
    assert list(node_left_siblings((0,))) == []

def node_left_siblings_2_test():
    assert list(node_left_siblings((0, 0, 0))) == []

def node_left_siblings_3_test():
    assert list(node_left_siblings((0, 2, 0))) == []

def node_left_siblings_4_test():
    assert list(node_left_siblings((0, 2, 1))) == [(0, 2, 0)]

def node_left_siblings_5_test():
    assert list(node_left_siblings((0, 2, 3))) == [(0, 2, 2), (0, 2, 1), (0, 2, 0)]


def tree_iter_1_test():
    assert list(tree_iter((1,))) == [(0,)]

def tree_iter_2_test():
    assert list(tree_iter((1, 5))) == [(0,), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

def tree_iter_3_test():
    assert list(tree_iter((1, 5, 2))) == [(0,), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (0, 3, 0), (0, 3, 1), (0, 4, 0), (0, 4, 1)]


def tree_index_b_1_test():
    assert_array_equal(tree_index_b((1,)), [1])

def tree_index_b_2_test():
    assert_array_equal(tree_index_b((1, 5)), [1, 5])

def tree_index_b_3_test():
    assert_array_equal(tree_index_b((1, 10, 5)), [1, 10, 50])

def tree_index_b_4_test():
    assert_array_equal(tree_index_b((1, 20, 10, 5)), [1, 20, 200, 1000])


def tree_index_m_1_test():
    assert_array_equal(tree_index_m((1,)), [[1]])

def tree_index_m_2_test():
    assert_array_equal(tree_index_m((1, 5)), [[1, 0], [5, 1]])

def tree_index_m_3_test():
    assert_array_equal(tree_index_m((1, 10, 5)),
        [[ 1,  0,  0],
         [10,  1,  0],
         [50,  5,  1]])

def tree_index_m_4_test():
    assert_array_equal(tree_index_m((1, 20, 10, 5)),
        [[   1,    0,    0,    0],
         [  20,    1,    0,    0],
         [ 200,   10,    1,    0],
         [1000,   50,    5,    1]])


def _tree_index_m_b():
    t = (1, 20, 10, 5)
    m = tree_index_m(t)
    b = tree_index_b(t)
    return (m, b)

def tree_index_1_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0], m, b) == 0

def tree_index_2_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 0], m, b) == 1

def tree_index_3_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 19], m, b) == 20

def tree_index_4_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 0, 0], m, b) == 21

def tree_index_5_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 0, 9], m, b) == 30

def tree_index_6_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 1, 0], m, b) == 31

def tree_index_7_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 1, 9], m, b) == 40

def tree_index_8_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 19, 0], m, b) == 211

def tree_index_9_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 19, 9], m, b) == 220

def tree_index_10_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 0, 0, 0], m, b) == 221

def tree_index_11_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 0, 0, 4], m, b) == 225

def tree_index_12_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 0, 1, 0], m, b) == 226

def tree_index_13_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 0, 1, 4], m, b) == 230

def tree_index_14_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 0, 9, 0], m, b) == 266

def tree_index_15_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 0, 9, 4], m, b) == 270

def tree_index_16_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 1, 0, 0], m, b) == 271

def tree_index_17_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 1, 0, 4], m, b) == 275

def tree_index_18_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 1, 1, 0], m, b) == 276

def tree_index_19_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 1, 1, 4], m, b) == 280

def tree_index_20_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 1, 9, 0], m, b) == 316

def tree_index_21_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 1, 9, 4], m, b) == 320

def tree_index_22_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 19, 0, 0], m, b) == 1171

def tree_index_23_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 19, 0, 4], m, b) == 1175

def tree_index_24_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 19, 1, 0], m, b) == 1176

def tree_index_25_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 19, 1, 4], m, b) == 1180

def tree_index_26_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 19, 9, 0], m, b) == 1216

def tree_index_27_test():
    (m, b) = _tree_index_m_b()
    assert tree_index([0, 19, 9, 4], m, b) == 1220


def subtree_node_candidates_1_test():
    assert list(subtree_node_candidates((1,), {(0,): (0,)})) == []

def subtree_node_candidates_2_test():
    assert list(subtree_node_candidates((1, 5), {(0,): (0,)})) == [((0, 0), (0, 0)), ((0, 0), (0, 1)), ((0, 0), (0, 2)), ((0, 0), (0, 3)), ((0, 0), (0, 4))]

def subtree_node_candidates_3_test():
    assert list(subtree_node_candidates((1, 5), {
        (0,): (0,),
        (0, 0): (0, 0),
        (0, 1): (0, 2)
    })) == [((0, 2), (0, 1)), ((0, 2), (0, 3)), ((0, 2), (0, 4))]

def subtree_node_candidates_4_test():
    assert list(subtree_node_candidates((1, 5), {
        (0,): (0,),
        (0, 0): (0, 4),
        (0, 1): (0, 2)
    })) == [((0, 2), (0, 0)), ((0, 2), (0, 1)), ((0, 2), (0, 3))]

def subtree_node_candidates_5_test():
    assert list(subtree_node_candidates((1, 5), {
        (0,): (0,),
        (0, 0): (0, 4),
        (0, 1): (0, 2),
        (0, 2): (0, 0),
        (0, 3): (0, 1),
        (0, 4): (0, 2)
    })) == []

def subtree_node_candidates_6_test():
    assert list(subtree_node_candidates((1, 4, 3), {
        (0,): (0,),
        (0, 0): (0, 2),
        (0, 1): (0, 0)
    })) == [((0, 2), (0, 1)), ((0, 2), (0, 3)), ((0, 0, 0), (0, 2, 0)), ((0, 0, 0), (0, 2, 1)), ((0, 0, 0), (0, 2, 2)), ((0, 1, 0), (0, 0, 0)), ((0, 1, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 2))]

def subtree_node_candidates_7_test():
    assert list(subtree_node_candidates((1, 4, 3), {
        (0,): (0,),
        (0, 0): (0, 2),
        (0, 1): (0, 0),
        (0, 1, 0): (0, 0, 1)
    })) == [((0, 2), (0, 1)), ((0, 2), (0, 3)), ((0, 0, 0), (0, 2, 0)), ((0, 0, 0), (0, 2, 1)), ((0, 0, 0), (0, 2, 2)), ((0, 1, 1), (0, 0, 0)), ((0, 1, 1), (0, 0, 2))]

def subtree_node_candidates_8_test():
    assert list(subtree_node_candidates((1, 4, 3), {
        (0,): (0,),
        (0, 0): (0, 2),
        (0, 1): (0, 0),
        (0, 1, 0): (0, 0, 1),
        (0, 0, 0): (0, 2, 0)
    })) == [((0, 2), (0, 1)), ((0, 2), (0, 3)), ((0, 0, 1), (0, 2, 1)), ((0, 0, 1), (0, 2, 2)), ((0, 1, 1), (0, 0, 0)), ((0, 1, 1), (0, 0, 2))]

def subtree_node_candidates_9_test():
    assert list(subtree_node_candidates((1, 4, 3), {
        (0,): (0,),
        (0, 0): (0, 2),
        (0, 1): (0, 0),
        (0, 1, 0): (0, 0, 1),
        (0, 0, 0): (0, 2, 0),
        (0, 0, 1): (0, 2, 2),
        (0, 0, 2): (0, 2, 1)
    })) == [((0, 2), (0, 1)), ((0, 2), (0, 3)), ((0, 1, 1), (0, 0, 0)), ((0, 1, 1), (0, 0, 2))]

def subtree_node_candidates_10_test():
    assert list(subtree_node_candidates((1, 4, 3), {
        (0,): (0,),
        (0, 0): (0, 2),
        (0, 1): (0, 0),
        (0, 2): (0, 1),
        (0, 3): (0, 3),
        (0, 1, 0): (0, 0, 1),
        (0, 0, 0): (0, 2, 0),
        (0, 0, 1): (0, 2, 2),
        (0, 0, 2): (0, 2, 1),
        (0, 2, 0): (0, 1, 0),
        (0, 2, 1): (0, 1, 1),
        (0, 2, 2): (0, 1, 2),
        (0, 3, 0): (0, 3, 0),
        (0, 3, 1): (0, 3, 2)
    })) == [((0, 1, 1), (0, 0, 0)), ((0, 1, 1), (0, 0, 2)), ((0, 3, 2), (0, 3, 1))]

def subtree_node_candidates_11_test():
    assert list(subtree_node_candidates((1, 4, 3), {
        (0,): (0,),
        (0, 0): (0, 2),
        (0, 1): (0, 0),
        (0, 2): (0, 1),
        (0, 3): (0, 3),
        (0, 1, 0): (0, 0, 1),
        (0, 1, 1): (0, 0, 0),
        (0, 1, 2): (0, 0, 2),
        (0, 0, 0): (0, 2, 0),
        (0, 0, 1): (0, 2, 2),
        (0, 0, 2): (0, 2, 1),
        (0, 2, 0): (0, 1, 0),
        (0, 2, 1): (0, 1, 1),
        (0, 2, 2): (0, 1, 2),
        (0, 3, 0): (0, 3, 0),
        (0, 3, 1): (0, 3, 2),
        (0, 3, 2): (0, 3, 1)
    })) == []


def _sort_clusters(assignments, means, norm=None, reverse=False):
    '''
    Relabel/reorder clusters from k-means in order of cluster mean norm.
    Test helper.
    '''

    ids = np.argsort(vector_norm(means, ord=norm, axis=1))
    if reverse:
        ids = means.shape[0] - (ids + 1)
    (perm, perm_new_assignments) = np.where(assignments[:, np.newaxis] == ids)
    return (perm_new_assignments[np.argsort(perm)], means[ids, :])


def sort_clusters_1_test():
    (assignments, means) = _sort_clusters(np.array([0]), np.array([[0.]]))
    assert_array_equal(assignments, [0])
    assert_array_almost_equal(means, [[0]])

def sort_clusters_2_test():
    (assignments, means) = _sort_clusters(np.array([0]), np.array([[0., -42., 7.]]))
    assert_array_equal(assignments, [0])
    assert_array_almost_equal(means, [[0, -42, 7]])

def sort_clusters_3_test():
    (assignments, means) = _sort_clusters(np.array([0, 0, 0, 0]), np.array([[0., -42., 7.]]))
    assert_array_equal(assignments, [0, 0, 0, 0])
    assert_array_almost_equal(means, [[0, -42, 7]])

def sort_clusters_4_test():
    (assignments, means) = _sort_clusters(np.array([0, 1, 1, 0]), np.array([
        [0., -42., 7.],
        [42., 4., 4.,],
    ]))
    assert_array_equal(assignments, [1, 0, 0, 1])
    assert_array_almost_equal(means,
        [[ 42.,   4.,   4.],
         [  0., -42.,   7.]])

def sort_clusters_5_test():
    (assignments, means) = _sort_clusters(np.array([0, 1, 1, 0]), np.array([
        [0., -42., 7.],
        [42., 4., 4.,],
    ]), reverse=True)
    assert_array_equal(assignments, [0, 1, 1, 0])
    assert_array_almost_equal(means,
        [[  0., -42.,   7.],
         [ 42.,   4.,   4.]])

def sort_clusters_6_test():
    (assignments, means) = _sort_clusters(np.array([0, 1, 1, 0]), np.array([
        [0., -42., 7.],
        [42., 4., 4.,],
    ]), norm=1)
    assert_array_equal(assignments, [0, 1, 1, 0])
    assert_array_almost_equal(means,
        [[  0., -42.,   7.],
         [ 42.,   4.,   4.]])

def sort_clusters_7_test():
    (assignments, means) = _sort_clusters(np.array([0, 1, 1, 0]), np.array([
        [0., -42., 7.],
        [42., 4., 4.,],
    ]), reverse=True, norm=1)
    assert_array_equal(assignments, [1, 0, 0, 1])
    assert_array_almost_equal(means,
        [[ 42.,   4.,   4.],
         [  0., -42.,   7.]])

def sort_clusters_8_test():
    (assignments, means) = _sort_clusters(np.array([2, 0, 0, 2, 1, 1, 3]),
    np.array([
        [3., 3., 3.],
        [0., -42., 7.],
        [-42., 0., -4.,],
        [42., 4., 4.,],
    ]))
    assert_array_equal(assignments,
           [1, 0, 0, 1, 3, 3, 2])
    assert_array_almost_equal(means,
        [[  3.,   3.,   3.],
         [-42.,   0.,  -4.],
         [ 42.,   4.,   4.],
         [  0., -42.,   7.]])


def kmeans_1_test():
    (assignments, means) = _sort_clusters(*kmeans(np.array([[0]]), 1), norm=1)
    assert_array_equal(assignments, [0])
    assert_array_almost_equal(means, [[0]])

def kmeans_2_test():
    (assignments, means) = _sort_clusters(*kmeans(np.array([[0,0]]), 1), norm=1)
    assert_array_equal(assignments, [0])
    assert_array_almost_equal(means, [[0, 0]])

def kmeans_3_test():
    (assignments, means) = _sort_clusters(*kmeans(
        np.array([
            [0,0],
            [-3,5],
            [-5,4],
            [0,-1],
        ]),
        1
    ), norm=1)
    assert_array_equal(assignments, [0, 0, 0, 0])
    assert_array_almost_equal(means, [[-2, 2]])

def kmeans_4_test():
    (assignments, means) = _sort_clusters(*kmeans(
        np.array([
            [0,0.5],
            [-3,5],
            [-5,4],
            [0,-1],
        ]),
        7
    ), norm=1)
    assert_array_equal(assignments, [3, 5, 6, 4])
    assert_array_almost_equal(means, 
        [[ 0. ,  0. ],
         [ 0. ,  0. ],
         [ 0. ,  0. ],
         [ 0. ,  0.5],
         [ 0. , -1. ],
         [-3. ,  5. ],
         [-5. ,  4. ]])

def kmeans_5_test():
    (assignments, means) = _sort_clusters(*kmeans(
        np.array([
            [0,0],
            [-7,5],
            [-9,4],
            [0,-1],
        ]),
        2
    ), norm=1)
    assert_array_equal(assignments, [0, 1, 1, 0])
    assert_array_almost_equal(means, 
        [[ 0. , -0.5],
         [-8. ,  4.5]])

def kmeans_6_test():
    (assignments, means) = _sort_clusters(*kmeans(
        np.array([
            [0,0,2],
            [-7,5,-3],
            [-9,4,-3],
            [0,-1,0],
        ]),
        2
    ), norm=1)
    assert_array_equal(assignments, [0, 1, 1, 0])
    assert_array_almost_equal(means, 
        [[ 0. , -0.5,  1. ],
         [-8. ,  4.5, -3. ]])

def kmeans_7_test():
    (assignments, means) = _sort_clusters(*kmeans(
        np.array([
            [0,0,2],
            [-7,5,-3],
            [-9,4,-3],
            [99,-1,0],
            [0,-1,0],
        ]),
        3
    ), norm=1)
    assert_array_equal(assignments, [0, 1, 1, 2, 0])
    assert_array_almost_equal(means, 
        [[  0. ,  -0.5,   1. ],
         [ -8. ,   4.5,  -3. ],
         [ 99. ,  -1. ,   0. ]])


def _full_to_kmeans_sparse(data):
    '''
    Convert full k-means data format (N x D matrix) to list of tuples
    representing the rows.  The first element of a tuple is a list of
    feature ids, the second element is a corresponding vector of values.
    Return a pair containing this list of tuples and the total number of
    features.
    Test helper.
    '''

    sparse_data = []
    for i in xrange(data.shape[0]):
        feature_ids = np.where(data[i,:] != 0)[0]
        feature_values = data[i,feature_ids]
        sparse_data.append((feature_ids, feature_values))

    return (sparse_data, data.shape[1])


def full_to_kmeans_sparse_1_test():
    (data, num_features) = _full_to_kmeans_sparse(np.array([[0]]))
    assert len(data) == 1
    assert len(data[0][0]) == 0
    assert len(data[0][1]) == 0
    assert num_features == 1

def full_to_kmeans_sparse_2_test():
    (data, num_features) = _full_to_kmeans_sparse(np.array([[0,0]]))
    assert len(data) == 1
    assert len(data[0][0]) == 0
    assert len(data[0][1]) == 0
    assert num_features == 2

def full_to_kmeans_sparse_3_test():
    (data, num_features) = _full_to_kmeans_sparse(np.array([
        [0,0],
        [-3,5],
        [-5,4],
        [0,-1],
    ]))
    assert len(data[0][0]) == 0
    assert len(data[0][1]) == 0
    assert_array_equal(data[1], [[0, 1], [-3,  5]])
    assert_array_equal(data[2], [[0, 1], [-5,  4]])
    assert_array_equal(data[3], [[1], [-1]])
    assert num_features == 2

def full_to_kmeans_sparse_4_test():
    (data, num_features) = _full_to_kmeans_sparse(np.array([
        [0,0,2],
        [-7,5,-3],
        [-9,4,-3],
        [0,-1,0],
    ]))
    assert_array_equal(data[0], [[2], [2]])
    assert_array_equal(data[1], [[0, 1, 2], [-7, 5, -3]])
    assert_array_equal(data[2], [[0, 1, 2], [-9, 4, -3]])
    assert_array_equal(data[3], [[1], [-1]])
    assert num_features == 3


def kmeans_sparse_1_test():
    (data, num_features) = _full_to_kmeans_sparse(np.array([[0]]))
    (assignments, means) = _sort_clusters(*kmeans_sparse(data, num_features, 1), norm=1)
    assert_array_equal(assignments, [0])
    assert_array_almost_equal(means, [[0]])

def kmeans_sparse_2_test():
    (data, num_features) = _full_to_kmeans_sparse(np.array([[0,0]]))
    (assignments, means) = _sort_clusters(*kmeans_sparse(data, num_features, 1), norm=1)
    assert_array_equal(assignments, [0])
    assert_array_almost_equal(means, [[0, 0]])

def kmeans_sparse_3_test():
    (data, num_features) = _full_to_kmeans_sparse(np.array([
        [0,0],
        [-3,5],
        [-5,4],
        [0,-1],
    ]))
    (assignments, means) = _sort_clusters(*kmeans_sparse(data, num_features, 1), norm=1)
    assert_array_equal(assignments, [0, 0, 0, 0])
    assert_array_almost_equal(means, [[-2, 2]])

def kmeans_sparse_4_test():
    (data, num_features) = _full_to_kmeans_sparse(np.array([
        [0,0.5],
        [-3,5],
        [-5,4],
        [0,-1],
    ]))
    (assignments, means) = _sort_clusters(*kmeans_sparse(data, num_features, 7), norm=1)
    assert_array_equal(assignments, [3, 5, 6, 4])
    assert_array_almost_equal(means,
        [[ 0. ,  0. ],
         [ 0. ,  0. ],
         [ 0. ,  0. ],
         [ 0. ,  0.5],
         [ 0. , -1. ],
         [-3. ,  5. ],
         [-5. ,  4. ]])

def kmeans_sparse_5_test():
    (data, num_features) = _full_to_kmeans_sparse(np.array([
        [0,0],
        [-7,5],
        [-9,4],
        [0,-1],
    ]))
    (assignments, means) = _sort_clusters(*kmeans_sparse(data, num_features, 2), norm=1)
    assert_array_equal(assignments, [0, 1, 1, 0])
    assert_array_almost_equal(means,
        [[ 0. , -0.5],
         [-8. ,  4.5]])

def kmeans_sparse_6_test():
    (data, num_features) = _full_to_kmeans_sparse(np.array([
        [0,0,2],
        [-7,5,-3],
        [-9,4,-3],
        [0,-1,0],
    ]))
    (assignments, means) = _sort_clusters(*kmeans_sparse(data, num_features, 2), norm=1)
    assert_array_equal(assignments, [0, 1, 1, 0])
    assert_array_almost_equal(means,
        [[ 0. , -0.5,  1. ],
         [-8. ,  4.5, -3. ]])

def kmeans_sparse_7_test():
    (data, num_features) = _full_to_kmeans_sparse(np.array([
        [0,0,2],
        [-7,5,-3],
        [-9,4,-3],
        [99,-1,0],
        [0,-1,0],
    ]))
    (assignments, means) = _sort_clusters(*kmeans_sparse(data, num_features, 3), norm=1)
    assert_array_equal(assignments, [0, 1, 1, 2, 0])
    assert_array_almost_equal(means,
        [[  0. ,  -0.5,   1. ],
         [ -8. ,   4.5,  -3. ],
         [ 99. ,  -1. ,   0. ]])
