import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_array_almost_equal

from corpus import Document
from m0 import *


def update_ss_stochastic_1_t_0_trivial_test():
    iota = 0.
    kappa = 0.5

    onhdp = model((1,), D=1, W=1, iota=iota, kappa=kappa)
    onhdp.m_t = 0
    onhdp.m_tau_ss = np.zeros((2,1))
    onhdp.m_lambda_ss = np.zeros((1,1))
    onhdp.m_lambda_ss_sum = np.sum(onhdp.m_lambda_ss, axis=1)

    batch_to_vocab_word_map = [0]
    ss = suff_stats(1, 1, 1)

    onhdp.update_ss_stochastic(ss, batch_to_vocab_word_map)

    assert_array_almost_equal(onhdp.m_tau_ss, np.zeros((2,1)))
    assert_array_almost_equal(onhdp.m_lambda_ss, np.zeros((1,1)))
    assert_array_almost_equal(onhdp.m_lambda_ss_sum, np.sum(onhdp.m_lambda_ss, axis=1))

def update_ss_stochastic_1_t_0_non_trivial_test():
    iota = 2.
    kappa = 0.5

    onhdp = model((1,), D=4, W=3, iota=iota, kappa=kappa)
    onhdp.m_t = 0
    onhdp.m_tau_ss = np.array([[1], [0]])
    onhdp.m_lambda_ss = np.array([[0.25, 4, 0.5]])
    onhdp.m_lambda_ss_sum = np.sum(onhdp.m_lambda_ss, axis=1)

    batch_to_vocab_word_map = [2, 0]
    ss = suff_stats(1, 2, 3)
    ss.m_tau_ss[:] = np.array([42])
    ss.m_lambda_ss[:] = np.array([[9, 72]])

    onhdp.update_ss_stochastic(ss, batch_to_vocab_word_map)

    assert_array_almost_equal(onhdp.m_tau_ss, np.array(
        [[32.754264805429422], [32.331615074619044]]))
    assert_array_almost_equal(onhdp.m_lambda_ss, np.array(
        [[55.531288274906672, 1.6905989232414971, 7.1395280956806966]]))
    assert_array_almost_equal(onhdp.m_lambda_ss_sum, np.sum(onhdp.m_lambda_ss, axis=1))

def update_ss_stochastic_1_t_3_non_trivial_test():
    iota = 2.
    kappa = 0.5

    onhdp = model((1,), D=4, W=3, iota=iota, kappa=kappa)
    onhdp.m_t = 3
    onhdp.m_tau_ss = np.array([[1], [0]])
    onhdp.m_lambda_ss = np.array([[0.25, 4, 0.5]])
    onhdp.m_lambda_ss_sum = np.sum(onhdp.m_lambda_ss, axis=1)

    batch_to_vocab_word_map = [2, 0]
    ss = suff_stats(1, 2, 3)
    ss.m_tau_ss[:] = np.array([42])
    ss.m_lambda_ss[:] = np.array([[9, 72]])

    onhdp.update_ss_stochastic(ss, batch_to_vocab_word_map)

    assert_array_almost_equal(onhdp.m_tau_ss, np.array(
        [[23.453655975512469], [22.861904265976332]]))
    assert_array_almost_equal(onhdp.m_lambda_ss, np.array(
        [[39.339773811914888, 2.3670068381445479, 5.1948553403344251]]))
    assert_array_almost_equal(onhdp.m_lambda_ss_sum, np.sum(onhdp.m_lambda_ss, axis=1))

def update_ss_stochastic_1_2_2_t_3_non_trivial_test():
    iota = 2.
    kappa = 0.5

    onhdp = model((1,2,2), D=4, W=3, iota=iota, kappa=kappa)
    onhdp.m_t = 3
    onhdp.m_tau_ss = np.array([[1, 4, 1, 3.5, 1, 7, 1], [0, 8, 0, 8, 0, 2, 0]])
    onhdp.m_lambda_ss = np.array(
        [[0.25, 4, 0.5],
         [3.25, 3, 3.5],
         [3, 3, 3],
         [1, 0, 1],
         [1, 1, 1],
         [12, 42, 72],
         [1, 2, 3]])
    onhdp.m_lambda_ss_sum = np.sum(onhdp.m_lambda_ss, axis=1)

    batch_to_vocab_word_map = [2, 0]
    ss = suff_stats(7, 2, 3)
    ss.m_tau_ss[:] = np.array([42, 42, 47, 0, 8, 8.5, 7.5])
    ss.m_lambda_ss[:] = np.array(
        [[9, 2.5],
         [0, 72.5],
         [0, 73.5],
         [1, 3.5],
         [2, 8],
         [3, 3.5],
         [4, 4]])

    onhdp.update_ss_stochastic(ss, batch_to_vocab_word_map)

    assert_array_almost_equal(onhdp.m_tau_ss, np.array(
        [[ 23.45365598,  25.2289111 ,  26.17531125,   2.07113098,
            4.94640014,   8.76907593,   4.67423461],
         [ 22.86190427,  27.59591794,  25.58355954,   4.73401368,
            4.35464843,   5.81031738,   4.0824829 ]]))
    #assert_array_almost_equal(onhdp.m_lambda_ss, np.array(
    #assert_array_almost_equal(onhdp.m_lambda_ss_sum, np.sum(onhdp.m_lambda_ss, axis=1))


def update_lambda_1_one_type_test():
    onhdp = model((1,), D=47, W=1, lambda0=2)
    onhdp.m_lambda_ss = np.ones((1, 1))
    onhdp.m_lambda_ss_sum = np.sum(onhdp.m_lambda_ss, axis=1)
    onhdp.update_lambda()
    assert_array_almost_equal(onhdp.m_Elogprobw, np.zeros((1,1)))

def update_lambda_1_many_types_test():
    onhdp = model((1,), D=47, W=3, lambda0=2)
    onhdp.m_lambda_ss = np.array([[3, 0, 2.5]])
    onhdp.m_lambda_ss_sum = np.sum(onhdp.m_lambda_ss, axis=1)
    onhdp.update_lambda()
    assert_array_almost_equal(onhdp.m_Elogprobw,
        np.array([[-0.89212146, -1.97545479, -1.0093682 ]]))

def update_lambda_1_2_2_one_type_test():
    onhdp = model((1,2,2), D=47, W=1, lambda0=2)
    onhdp.m_lambda_ss = np.ones((7, 1))
    onhdp.m_lambda_ss_sum = np.sum(onhdp.m_lambda_ss, axis=1)
    onhdp.update_lambda()
    assert_array_almost_equal(onhdp.m_Elogprobw, np.zeros((7,1)))

def update_lambda_1_2_2_many_types_test():
    onhdp = model((1,2,2), D=47, W=3, lambda0=2)
    onhdp.m_lambda_ss = np.array(
        [[3, 0, 2.5],
         [3, 0, 2.5],
         [1, 0, 2],
         [1, 0, 4.5],
         [3, 3, 0.5],
         [0, 0, 0],
         [3, 42, 2.5]])
    onhdp.m_lambda_ss_sum = np.sum(onhdp.m_lambda_ss, axis=1)
    onhdp.update_lambda()
    assert_array_almost_equal(onhdp.m_Elogprobw, np.array(
        [[-0.89212146, -1.97545479, -1.0093682 ],
         [-0.89212146, -1.97545479, -1.0093682 ],
         [-1.21785714, -1.71785714, -0.88452381],
         [-1.47545479, -1.97545479, -0.6053278 ],
         [-0.97907798, -0.97907798, -1.78203901],
         [-1.28333333, -1.28333333, -1.28333333],
         [-2.46418908, -0.19752379, -2.58143582]]))


def update_nu_1_one_type_test():
    onhdp = model((1,), D=47, W=42)
    doc = Document([3], [2])
    ab = np.array([[1], [0]])
    uv = np.array([[1], [0]])
    nu = np.ones((2,1))
    log_nu = np.log(nu)
    Elogprobw_doc = np.zeros((1,1))
    onhdp.update_nu(
        {(0,): (0,)},
        ab,
        uv,
        Elogprobw_doc,
        doc,
        nu,
        log_nu)
    assert_array_almost_equal(nu, [[1], [1]])
    assert_array_almost_equal(log_nu, np.log(nu))

def update_nu_1_many_types_test():
    onhdp = model((1,), D=47, W=42)
    doc = Document([3, 8, 0], [2, 1, 3])
    ab = np.array([[1], [0]])
    uv = np.array([[1], [0]])
    nu = np.ones((6,1))
    log_nu = np.log(nu)
    Elogprobw_doc = np.log(np.array([[1., 1., 1.]]) / 3)
    onhdp.update_nu(
        {(0,): (0,)},
        ab,
        uv,
        Elogprobw_doc,
        doc,
        nu,
        log_nu)
    assert_array_almost_equal(nu, [[1], [1], [1], [1], [1], [1]])
    assert_array_almost_equal(log_nu, np.log(nu))

def update_nu_1_3_one_type_full_subtree_test():
    onhdp = model((1, 3), D=47, W=42, gamma1=2, gamma2=3, alpha=4, beta=5)
    doc = Document([3], [2])
    ab = np.array([[8, 1, 1, 1], [4, 0, 0, 0]])
    uv = np.array([[1, 5, 2, 1], [0, 6, 7, 0]])
    nu = np.ones((2,4))/4.
    log_nu = np.log(nu)
    Elogprobw_doc = np.zeros((4,1))
    onhdp.update_nu(
        {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1), (0, 2): (0, 0)},
        ab,
        uv,
        Elogprobw_doc,
        doc,
        nu,
        log_nu)
    assert_array_almost_equal(nu,
        [[0.69805004, 0.1402087, 0.03073109, 0.13101016],
         [0.69805004, 0.1402087, 0.03073109, 0.13101016]])
    assert_array_almost_equal(log_nu, np.log(nu))

def update_nu_1_3_many_types_full_subtree_test():
    onhdp = model((1, 3), D=47, W=42, gamma1=2, gamma2=3, alpha=4, beta=5)
    doc = Document([3, 8, 0], [2, 1, 3])
    ab = np.array([[8, 1, 1, 1], [4, 0, 0, 0]])
    uv = np.array([[1, 5, 2, 1], [0, 6, 7, 0]])
    nu = np.ones((6,4))/4.
    log_nu = np.log(nu)
    Elogprobw_doc = np.log(np.array(
        [[0.25, 0.5, 0.25],
         [0.125, 0.625, 0.25],
         [0.25, 0.5, 0.25],
         [0.0625, 0.0625, 0.875]]))
    onhdp.update_nu(
        {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1), (0, 2): (0, 0)},
        ab,
        uv,
        Elogprobw_doc,
        doc,
        nu,
        log_nu)
    assert_array_almost_equal(nu, np.array(
        [[ 0.83936763,  0.08429671,  0.03695249,  0.03938317],
         [ 0.83936763,  0.08429671,  0.03695249,  0.03938317],
         [ 0.75840523,  0.19041438,  0.03338818,  0.0177922 ],
         [ 0.525828  ,  0.10561659,  0.02314916,  0.34540625],
         [ 0.525828  ,  0.10561659,  0.02314916,  0.34540625],
         [ 0.525828  ,  0.10561659,  0.02314916,  0.34540625]]))
    assert_array_almost_equal(log_nu, np.log(nu))

def update_nu_1_3_many_types_partial_subtree_test():
    onhdp = model((1, 3), D=47, W=42, gamma1=2, gamma2=3, alpha=4, beta=5)
    doc = Document([3, 8, 0], [2, 1, 3])
    ab = np.array([[8, 1, 1, 1], [4, 0, 0, 0]])
    uv = np.array([[1, 5, 1, 1], [0, 6, 0, 0]])
    nu = np.ones((6,4))/4.
    log_nu = np.log(nu)
    Elogprobw_doc = np.log(np.array(
        [[0.25, 0.5, 0.25],
         [0.125, 0.625, 0.25],
         [0.25, 0.5, 0.25],
         [0.0625, 0.0625, 0.875]]))
    onhdp.update_nu(
        {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1)},
        ab,
        uv,
        Elogprobw_doc,
        doc,
        nu,
        log_nu)
    assert_array_almost_equal(nu, np.array(
        [[ 0.74307623,  0.07462628,  0.18229749,  0.        ],
         [ 0.74307623,  0.07462628,  0.18229749,  0.        ],
         [ 0.66827043,  0.16778405,  0.16394552,  0.        ],
         [ 0.69147409,  0.13888788,  0.16963803,  0.        ],
         [ 0.69147409,  0.13888788,  0.16963803,  0.        ],
         [ 0.69147409,  0.13888788,  0.16963803,  0.        ]]))
    assert_array_almost_equal(log_nu, np.log(nu))

def update_nu_1_2_2_one_type_full_subtree_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=2, gamma2=3, alpha=4, beta=5)
    doc = Document([3], [2])
    ab = np.array([[8, 3, 3, 1, 1, 1, 1], [4, 4, 5, 0, 0, 0, 0]])
    uv = np.array([[1, 5, 1, 2, 1, 3, 1], [0, 6, 0, 7, 0, 8, 0]])
    nu = np.ones((2,7))/7.
    log_nu = np.log(nu)
    Elogprobw_doc = np.zeros((7,1))
    onhdp.update_nu(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        ab,
        uv,
        Elogprobw_doc,
        doc,
        nu,
        log_nu)
    assert_array_almost_equal(nu, np.array(
        [[ 0.71278782,  0.05536929,  0.05862539,  0.01386685,  0.05911598,
           0.02516698,  0.07506768],
         [ 0.71278782,  0.05536929,  0.05862539,  0.01386685,  0.05911598,
           0.02516698,  0.07506768]]))
    assert_array_almost_equal(log_nu, np.log(nu))

def update_nu_1_2_2_many_types_full_subtree_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=2, gamma2=3, alpha=4, beta=5)
    doc = Document([3, 8, 0], [2, 1, 3])
    ab = np.array([[8, 3, 3, 1, 1, 1, 1], [4, 4, 5, 0, 0, 0, 0]])
    uv = np.array([[1, 5, 1, 2, 1, 3, 1], [0, 6, 0, 7, 0, 8, 0]])
    nu = np.ones((6,7))/7.
    log_nu = np.log(nu)
    Elogprobw_doc = np.log(np.array(
        [[0.25, 0.5, 0.25],
         [0.125, 0.625, 0.25],
         [0.25, 0.5, 0.25],
         [0.25, 0.125, 0.25],
         [0.25, 0.125, 0.25],
         [0.75, 0.125, 0.125],
         [0.0625, 0.0625, 0.875]]))
    onhdp.update_nu(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        ab,
        uv,
        Elogprobw_doc,
        doc,
        nu,
        log_nu)
    assert_array_almost_equal(nu, np.array(
        [[ 0.73760944,  0.02864871,  0.06066692,  0.01434974,  0.06117459,
           0.07813015,  0.01942045],
         [ 0.73760944,  0.02864871,  0.06066692,  0.01434974,  0.06117459,
           0.07813015,  0.01942045],
         [ 0.81503779,  0.07914007,  0.06703525,  0.00396402,  0.01689905,
           0.0071943 ,  0.01072952],
         [ 0.60658368,  0.04711936,  0.04989031,  0.01180072,  0.0503078 ,
           0.01070857,  0.22358955],
         [ 0.60658368,  0.04711936,  0.04989031,  0.01180072,  0.0503078 ,
           0.01070857,  0.22358955],
         [ 0.60658368,  0.04711936,  0.04989031,  0.01180072,  0.0503078 ,
           0.01070857,  0.22358955]]))
    assert_array_almost_equal(log_nu, np.log(nu))

def update_nu_1_2_2_many_types_partial_subtree_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=2, gamma2=3, alpha=4, beta=5)
    doc = Document([3, 8, 0], [2, 1, 3])
    ab = np.array([[8, 1, 3, 1, 1, 1, 1], [4, 0, 5, 0, 0, 0, 0]])
    uv = np.array([[1, 5, 1, 1, 1, 1, 1], [0, 6, 0, 0, 0, 0, 0]])
    nu = np.ones((6,7))/7.
    log_nu = np.log(nu)
    Elogprobw_doc = np.log(np.array(
        [[0.25, 0.5, 0.25],
         [0.125, 0.625, 0.25],
         [0.25, 0.5, 0.25],
         [0.25, 0.125, 0.25],
         [0.25, 0.125, 0.25],
         [0.75, 0.125, 0.125],
         [0.0625, 0.0625, 0.875]]))
    onhdp.update_nu(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 1, 0): (0, 1, 1)},
        ab,
        uv,
        Elogprobw_doc,
        doc,
        nu,
        log_nu)
    assert_array_almost_equal(nu, np.array(
        [[ 0.61544419,  0.06180834,  0.05061907,  0.        ,  0.        ,
           0.2721284 ,  0.        ],
         [ 0.61544419,  0.06180834,  0.05061907,  0.        ,  0.        ,
           0.2721284 ,  0.        ],
         [ 0.7298379 ,  0.18324192,  0.06002773,  0.        ,  0.        ,
           0.02689245,  0.        ],
         [ 0.7370283 ,  0.14803779,  0.06061912,  0.        ,  0.        ,
           0.05431479,  0.        ],
         [ 0.7370283 ,  0.14803779,  0.06061912,  0.        ,  0.        ,
           0.05431479,  0.        ],
         [ 0.7370283 ,  0.14803779,  0.06061912,  0.        ,  0.        ,
           0.05431479,  0.        ]]))
    assert_array_almost_equal(log_nu, np.log(nu))


def update_uv_1_test():
    onhdp = model((1,), D=47, W=42, beta=9)
    uv = np.zeros((2,1))
    onhdp.update_uv(
        {(0,): (0,)},
        np.zeros((1,)),
        uv)
    assert_array_almost_equal(uv, [[1], [0]])

def update_uv_2_test():
    onhdp = model((1,), D=47, W=42, beta=9)
    uv = np.zeros((2,1))
    onhdp.update_uv(
        {(0,): (0,)},
        np.array([5]),
        uv)
    assert_array_almost_equal(uv, [[1], [0]])

def update_uv_3_test():
    onhdp = model((1, 3), D=47, W=42, beta=9)
    uv = np.zeros((2,4))
    onhdp.update_uv(
        {(0,): (0,)},
        np.array([0, 0, 0, 0]),
        uv)
    assert_array_almost_equal(uv[:,[0]], [[1], [0]])

def update_uv_4_test():
    onhdp = model((1, 3), D=47, W=42, beta=9)
    uv = np.zeros((2,4))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1), (0, 2): (0, 0)},
        np.array([0, 0, 0, 0]),
        uv)
    assert_array_almost_equal(uv,
        [[ 1., 1., 1., 1. ],
         [ 0., 9., 9., 0. ]])

def update_uv_5_test():
    onhdp = model((1, 3), D=47, W=42, beta=9)
    uv = np.zeros((2,4))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1), (0, 2): (0, 0)},
        np.array([2.5, 0.25, 4, 6]),
        uv)
    assert_array_almost_equal(uv,
        [[ 1, 1.25, 5., 1. ],
         [ 0, 19., 15., 0. ]])

def update_uv_6_test():
    onhdp = model((1, 3), D=47, W=42, beta=9)
    uv = np.zeros((2,4))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1)},
        np.array([2.5, 0.25, 4, 0]),
        uv)
    assert_array_almost_equal(uv[:,:3],
       [[ 1, 1.25, 1. ],
        [ 0, 13., 0. ]])

def update_uv_7_test():
    onhdp = model((1, 3), D=47, W=42, beta=9)
    uv = np.zeros((2,4))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1)},
        np.array([2.5, 0.25, 4, 6]), # degenerate case
        uv)
    assert_array_almost_equal(uv[:,:3],
       [[ 1, 1.25, 1. ],
        [ 0, 13., 0. ]])

def update_uv_8_test():
    onhdp = model((1, 2, 2), D=47, W=42, beta=9)
    uv = np.zeros((2,7))
    onhdp.update_uv(
        {(0,): (0,)},
        np.array([0, 0, 0, 0, 0, 0, 0]),
        uv)
    assert_array_almost_equal(uv[:,[0]], [[ 1. ], [ 0. ]])

def update_uv_9_test():
    onhdp = model((1, 2, 2), D=47, W=42, beta=9)
    uv = np.zeros((2,7))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([0, 0, 0, 0, 0, 0, 0]),
        uv)
    assert_array_almost_equal(uv,
       [[ 1., 1., 1., 1., 1., 1., 1. ],
         [ 0., 9., 0., 9., 0., 9., 0. ]])

def update_uv_10_test():
    onhdp = model((1, 2, 2), D=47, W=42, beta=9)
    uv = np.zeros((2,7))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([2.5, 0.25, 0, 6, 7, 0, 18]),
        uv)
    assert_array_almost_equal(uv,
       [[ 1, 14.25, 1., 7., 1., 1., 1. ],
        [ 0, 27., 0., 16., 0., 27., 0. ]])

def update_uv_11_test():
    onhdp = model((1, 2, 2), D=47, W=42, beta=9)
    uv = np.zeros((2,7))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([2.5, 0.25, 0, 6, 0, 0, 18]),
        uv)
    assert_array_almost_equal(uv[:,[0, 1, 2, 3, 5, 6]],
       [[ 1, 7.25, 1., 1., 1., 1. ],
        [ 0, 27., 0., 0., 27., 0. ]])

def update_uv_12_test():
    onhdp = model((1, 2, 2), D=47, W=42, beta=9)
    uv = np.zeros((2,7))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([2.5, 0.25, 0, 6, 7, 0, 18]), # degenerate case
        uv)
    assert_array_almost_equal(uv[:,[0, 1, 2, 3, 5, 6]],
       [[ 1, 7.25, 1., 1., 1., 1. ],
        [ 0, 27., 0., 0., 27., 0. ]])

def update_uv_13_test():
    onhdp = model((1, 2, 2), D=47, W=42, beta=9)
    uv = np.zeros((2,7))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([2.5, 0.25, 0, 0, 0, 0, 18]),
        uv)
    assert_array_almost_equal(uv[:,[0, 1, 2, 5, 6]],
       [[ 1, 1.25, 1., 1., 1. ],
        [ 0, 27., 0., 27., 0. ]])

def update_uv_14_test():
    onhdp = model((1, 2, 2), D=47, W=42, beta=9)
    uv = np.zeros((2,7))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([2.5, 0.25, 0, 6, 7, 0, 18]), # degenerate case
        uv)
    assert_array_almost_equal(uv[:,[0, 1, 2, 5, 6]],
       [[ 1, 1.25, 1., 1., 1. ],
        [ 0, 27., 0., 27., 0. ]])

def update_uv_15_test():
    onhdp = model((1, 2, 2), D=47, W=42, beta=9)
    uv = np.zeros((2,7))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0)},
        np.array([2.5, 0.25, 0, 6, 7, 0, 0]),
        uv)
    assert_array_almost_equal(uv[:,:5],
       [[ 1, 14.25, 1., 7., 1. ],
        [ 0, 9., 0., 16., 0. ]])

def update_uv_16_test():
    onhdp = model((1, 2, 2), D=47, W=42, beta=9)
    uv = np.zeros((2,7))
    onhdp.update_uv(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0)},
        np.array([2.5, 0.25, 0, 6, 7, 0, 18]), # degenerate case
        uv)
    assert_array_almost_equal(uv[:,:5],
       [[ 1, 14.25, 1., 7., 1. ],
        [ 0, 9., 0., 16., 0. ]])


def update_ab_1_test():
    onhdp = model((1,), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,1))
    onhdp.update_ab(
        {(0,): (0,)},
        np.zeros((1,)),
        ab)
    assert_array_almost_equal(ab, [[1], [0]])

def update_ab_2_test():
    onhdp = model((1,), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,1))
    onhdp.update_ab(
        {(0,): (0,)},
        np.array([5]),
        ab)
    assert_array_almost_equal(ab, [[ 1.], [ 0.]])

def update_ab_3_test():
    onhdp = model((1, 3), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,4))
    onhdp.update_ab(
        {(0,): (0,)},
        np.array([0, 0, 0, 0]),
        ab)
    assert_array_almost_equal(ab[:,[0]], [[ 1. ], [ 0. ]])

def update_ab_4_test():
    onhdp = model((1, 3), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,4))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1), (0, 2): (0, 0)},
        np.array([0, 0, 0, 0]),
        ab)
    assert_array_almost_equal(ab,
       [[ 9., 1., 1., 1. ],
        [ 3., 0., 0., 0. ]])

def update_ab_5_test():
    onhdp = model((1, 3), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,4))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1), (0, 2): (0, 0)},
        np.array([2.5, 0.25, 0, 6]),
        ab)
    assert_array_almost_equal(ab,
       [[ 11.5, 1., 1., 1. ],
        [ 9.25, 0., 0., 0. ]])

def update_ab_6_test():
    onhdp = model((1, 3), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,4))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1)},
        np.array([2.5, 0.25, 0, 0]),
        ab)
    assert_array_almost_equal(ab[:,:3],
       [[ 11.5, 1., 1. ],
        [ 3.25, 0., 0. ]])

def update_ab_7_test():
    onhdp = model((1, 3), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,4))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1)},
        np.array([2.5, 0.25, 0, 6]), # degenerate case
        ab)
    assert_array_almost_equal(ab[:,:3],
       [[ 11.5, 1., 1. ],
        [ 3.25, 0., 0. ]])

def update_ab_8_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,7))
    onhdp.update_ab(
        {(0,): (0,)},
        np.array([0, 0, 0, 0, 0, 0, 0]),
        ab)
    assert_array_almost_equal(ab[:,[0]],
       [[ 1. ],
        [ 0. ]])

def update_ab_9_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,7))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([0, 0, 0, 0, 0, 0, 0]),
        ab)
    assert_array_almost_equal(ab,
       [[ 9., 9., 9., 1., 1., 1., 1. ],
        [ 3., 3., 3., 0., 0., 0., 0. ]])

def update_ab_10_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,7))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([2.5, 0.25, 0, 6, 7, 0, 18]),
        ab)
    assert_array_almost_equal(ab,
       [[ 11.5, 9.25, 9., 1., 1., 1., 1. ],
        [ 34.25, 16., 21., 0., 0., 0., 0. ]])

def update_ab_11_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,7))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([2.5, 0.25, 0, 6, 0, 0, 18]),
        ab)
    assert_array_almost_equal(ab[:,[0, 1, 2, 3, 5, 6]],
       [[ 11.5, 9.25, 9., 1., 1., 1. ],
        [ 27.25, 9., 21., 0., 0., 0. ]])

def update_ab_12_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,7))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([2.5, 0.25, 0, 6, 7, 0, 18]), # degenerate case
        ab)
    assert_array_almost_equal(ab[:,[0, 1, 2, 3, 5, 6]],
       [[ 11.5, 9.25, 9., 1., 1., 1. ],
        [ 27.25, 9., 21., 0., 0., 0. ]])

def update_ab_13_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,7))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([2.5, 0.25, 0, 0, 0, 0, 18]),
        ab)
    assert_array_almost_equal(ab[:,[0, 1, 2, 5, 6]],
       [[ 11.5, 1, 9., 1., 1. ],
        [ 21.25, 0, 21., 0., 0. ]])

def update_ab_14_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,7))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
        np.array([2.5, 0.25, 0, 6, 7, 0, 18]), # degenerate case
        ab)
    assert_array_almost_equal(ab[:,[0, 1, 2, 5, 6]],
       [[ 11.5, 1, 9., 1., 1. ],
        [ 21.25, 0, 21., 0., 0. ]])

def update_ab_15_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,7))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0)},
        np.array([2.5, 0.25, 0, 6, 7, 0, 0]),
        ab)
    assert_array_almost_equal(ab[:,:5],
       [[ 11.5, 9.25, 1., 1., 1. ],
        [ 16.25, 16., 0., 0., 0. ]])

def update_ab_16_test():
    onhdp = model((1, 2, 2), D=47, W=42, gamma1=9, gamma2=3)
    ab = np.zeros((2,7))
    onhdp.update_ab(
        {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0)},
        np.array([2.5, 0.25, 0, 6, 7, 0, 18]), # degenerate case
        ab)
    assert_array_almost_equal(ab[:,:5],
       [[ 11.5, 9.25, 1., 1., 1. ],
        [ 16.25, 16., 0., 0., 0. ]])


def update_tau_1_test():
    onhdp = model((1,), D=47, W=42, alpha=9)
    onhdp.update_tau()
    assert_array_almost_equal(onhdp.m_tau, [[1], [0]])

def update_tau_2_test():
    onhdp = model((1, 3), D=47, W=42, alpha=9)
    onhdp.m_tau_ss[:] = 0
    onhdp.update_tau()
    assert_array_almost_equal(onhdp.m_tau,
       [[ 1.,  1.,  1.,  1.],
        [ 0.,  9.,  9.,  0.]])


def update_tau_3_test():
    onhdp = model((1, 3), D=47, W=42, alpha=9)
    onhdp.m_tau_ss[:] = [2, 5, 7, 3]
    onhdp.update_tau()
    assert_array_almost_equal(onhdp.m_tau,
       [[  1. ,   6,   8. ,   1. ],
        [  0. ,  19. ,  12. ,   0. ]])

def update_tau_4_test():
    onhdp = model((1, 2, 2), D=47, W=42, alpha=9)
    onhdp.m_tau_ss[:] = 0
    onhdp.update_tau()
    assert_array_almost_equal(onhdp.m_tau,
       [[ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 0.,  9.,  0.,  9.,  0.,  9.,  0.]])

def update_tau_5_test():
    onhdp = model((1, 2, 2), D=47, W=42, alpha=9)
    onhdp.m_tau_ss[:] = [2, 5, 7, 3, 2, 0, 14]
    onhdp.update_tau()
    assert_array_almost_equal(onhdp.m_tau,
       [[  1. ,   6,   1. ,   4. ,   1. ,   1. ,   1. ],
        [  0. ,  16. ,   0. ,  11. ,   0. ,  23. ,   0. ]])


def z_likelihood_1_test():
    onhdp = model((1,), D=47, W=42)
    assert_almost_equal(0,
        onhdp.z_likelihood(
            {(0,): (0,)},
            np.array([[0.], [np.inf]])
        ))

def z_likelihood_2_test():
    onhdp = model((1, 3), D=47, W=42)
    assert_almost_equal(0,
        onhdp.z_likelihood(
            {(0,): (0,)},
            np.array([[0., 0., 0., 0.], [np.inf, 0., 0., np.inf]])))

def z_likelihood_3_test():
    onhdp = model((1, 3), D=47, W=42)
    assert_almost_equal(-10.75,
        onhdp.z_likelihood(
            {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1), (0, 2): (0, 0)},
            np.array([[0., -0.5, -0.25, 0.], [np.inf, -1., -8., np.inf]])))

def z_likelihood_4_test():
    onhdp = model((1, 3), D=47, W=42)
    assert_almost_equal(-10.25,
        onhdp.z_likelihood(
            {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1)},
            np.array([[0., -0.5, -0.25, 0.], [np.inf, -1., -8., np.inf]])))

def z_likelihood_5_test():
    onhdp = model((1, 2, 2), D=47, W=42)
    assert_almost_equal(-73.875,
        onhdp.z_likelihood(
            {(0,): (0,), (0, 0): (0, 1), (0, 1): (0, 0), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0), (0, 1, 0): (0, 0, 0), (0, 1, 1): (0, 0, 1)},
            np.array([
                [0., -0.5, 0., -0.25, 0., -0.125, 0.],
                [np.inf, -1., np.inf, -8., np.inf, -64., np.inf]
            ])))

def z_likelihood_6_test():
    onhdp = model((1, 2, 2), D=47, W=42)
    assert_almost_equal(-65.125,
        onhdp.z_likelihood(
            {(0,): (0,), (0, 0): (0, 1), (0, 0, 0): (0, 1, 1), (0, 0, 1): (0, 1, 0)},
            np.array([
                [0., -0.5, 0., -0.25, 0., -0.125, 0.],
                [np.inf, -1., np.inf, -8., np.inf, -64., np.inf]
            ])))


def c_likelihood_1_test():
    onhdp = model((1,), D=47, W=42)
    assert_almost_equal(0,
        onhdp.c_likelihood(
            {(0,): (0,)},
            np.array([[1.], [0.]]),
            np.array([[1.], [0.]]),
            np.array([[1.], [1.], [1.]]),
            np.array([[0.], [0.], [0.]]),
            [0]))

def c_likelihood_2_test():
    onhdp = model((1, 3), D=47, W=42)
    nu = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.625, 0.125, 0.125, 0.125],
        [0.125, 0.125, 0.5, 0.25],
    ])
    assert_almost_equal(-3.200399838,
        onhdp.c_likelihood(
            {(0,): (0,), (0, 0): (0, 2), (0, 1): (0, 1), (0, 2): (0, 0)},
            np.array([[0.5, 1., 1., 1.], [1., 0., 0., 0.]]),
            np.array([[1., 2., 1., 1.], [0., 2., 4., 0.]]),
            nu,
            np.log(nu),
            [0, 1, 2, 3]))

def c_likelihood_3_test():
    onhdp = model((1, 3), D=47, W=42)
    nu = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.625, 0.125, 0.125, 0.125],
        [0.125, 0.125, 0.5, 0.25],
    ])
    assert_almost_equal(-3.200399838,
        onhdp.c_likelihood(
            {(0,): (0,), (0, 0): (0, 0), (0, 1): (0, 1), (0, 2): (0, 2)},
            np.array([[0.5, 1., 1., 1.], [1., 0., 0., 0.]]),
            np.array([[1., 2., 1., 1.], [0., 2., 4., 0.]]),
            nu,
            np.log(nu),
            [0, 1, 2, 3]))

def c_likelihood_4_test():
    onhdp = model((1, 3), D=47, W=42)
    nu = np.array([
        [0.25, 0.25, 0.5, 0.],
        [0.625, 0.25, 0.125, 0.],
        [0.125, 0.125, 0.75, 0.],
    ])
    assert_almost_equal(-2.218479183,
        onhdp.c_likelihood(
            {(0,): (0,), (0, 0): (0, 0), (0, 1): (0, 1)},
            np.array([[0.5, 1., 1., 1.], [1., 0., 0., 0.]]),
            np.array([[1., 2., 1., 1.], [0., 2., 0., 0.]]),
            nu,
            np.log(nu),
            [0, 1, 2]
        ))

def c_likelihood_5_test():
    onhdp = model((1, 2, 2), D=47, W=42)
    nu = np.array([
        [0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
        [0.0625, 0.0625, 0.0625, 0.625, 0.0625, 0.0625, 0.0625],
        [0.0625, 0.0625, 0.25, 0.25, 0.25, 0.0625, 0.0625],
    ])
    assert_almost_equal(-6.398696566,
        onhdp.c_likelihood(
            {(0,): (0,), (0, 0): (0, 0), (0, 1): (0, 1), (0, 0, 0): (0, 0, 0), (0, 0, 1): (0, 0, 1), (0, 1, 0): (0, 1, 0), (0, 1, 1): (0, 1, 1)},
            np.array([[0.5, 0.25, 0.125, 1., 1., 1., 1.], [1., 3., 2., 0., 0., 0., 0.]]),
            np.array([[1., 2., 1., 5., 1., 7., 1.], [0., 2., 0., 4., 0., 3., 0.]]),
            nu,
            np.log(nu),
            [0, 1, 2, 3, 4, 5, 6]))

def c_likelihood_6_test():
    onhdp = model((1, 2, 2), D=47, W=42)
    nu = np.array([
        [0.25, 0.25, 0., 0.25, 0.25, 0., 0.],
        [0.625, 0.125, 0., 0.125, 0.125, 0., 0.],
        [0.125, 0.125, 0., 0.25, 0.5, 0., 0.],
    ])
    assert_almost_equal(-3.488634305,
        onhdp.c_likelihood(
            {(0,): (0,), (0, 0): (0, 0), (0, 0, 0): (0, 0, 0), (0, 0, 1): (0, 0, 1)},
            np.array([[0.5, 0.25, 1., 1., 1., 1., 1.], [1., 3., 0., 0., 0., 0., 0.]]),
            np.array([[1., 1., 1., 5., 1., 1., 1.], [0., 0., 0., 4., 0., 0., 0.]]),
            nu,
            np.log(nu),
            [0, 1, 3, 4]))


def w_likelihood_1_test():
    onhdp = model((1,), D=47, W=42)
    doc = Document([3], [2])
    assert_almost_equal(0,
        onhdp.w_likelihood(
            doc,
            np.array([[1.], [1.]]),
            np.log(np.array([[1.]])),
            [0],
        ))

def w_likelihood_2_test():
    onhdp = model((1,), D=47, W=42)
    doc = Document([3], [2])
    assert_almost_equal(-4.158883083,
        onhdp.w_likelihood(
            doc,
            np.array([[1.], [1.]]),
            np.log(np.array([[0.125]])),
            [0]))

def w_likelihood_3_test():
    onhdp = model((1,), D=47, W=42)
    doc = Document([0, 41, 3], [2, 1, 2])
    assert_almost_equal(-6.238324625,
        onhdp.w_likelihood(
            doc,
        np.array([[1.], [1.], [1.], [1.], [1.]]),
        np.log(np.array([[0.25, 0.5, 0.25]])),
        [0]))

def w_likelihood_4_test():
    onhdp = model((1, 3), D=47, W=42)
    doc = Document([0, 41, 3], [2, 1, 2])
    assert_almost_equal(-11.911208475,
        onhdp.w_likelihood(
        doc,
        np.array([
            [0.125, 0.125, 0.125, 0.625],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.25, 0.125, 0.125],
            [0.125, 0.125, 0.5, 0.25],
            [0.0625, 0.0625, 0.125, 0.75],
        ]),
        np.log(np.array([
            [0.03125, 0.03125, 0.03125],
            [0.125, 0.75, 0.125],
            [0.25, 0.125, 0.25],
            [0.625, 0.0125, 0.0125],
        ])),
        [0, 1, 2, 3]))

def w_likelihood_5_test():
    onhdp = model((1, 3), D=47, W=42)
    doc = Document([0, 41, 3], [2, 1, 2])
    assert_almost_equal(-6.570175336,
        onhdp.w_likelihood(
        doc,
        np.array([
            [0.125, 0.125, 0.125, 0.625],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.25, 0.125, 0.125],
            [0.125, 0.125, 0.5, 0.25],
            [0.0625, 0.0625, 0.125, 0.75],
        ]),
        np.log(np.array([
            [0.03125, 0.03125, 0.03125],
            [0.125, 0.75, 0.125],
            [0.25, 0.125, 0.25],
            [0.625, 0.0125, 0.0125],
        ])),
        [0, 1, 2]))

def w_likelihood_6_test():
    onhdp = model((1, 3), D=47, W=42)
    doc = Document([0, 41, 3], [2, 1, 2])
    assert_almost_equal(-3.667621517,
        onhdp.w_likelihood(
        doc,
        np.array([
            [0.125, 0.125, 0.125, 0.625],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.25, 0.125, 0.125],
            [0.125, 0.125, 0.5, 0.25],
            [0.0625, 0.0625, 0.125, 0.75],
        ]),
        np.log(np.array([
            [0.25, 0.5, 0.25],
            [0.125, 0.75, 0.125],
            [0.5, 0.25, 0.25],
            [0.625, 0.125, 0.25],
        ])),
        [0, 1, 2]))

def w_likelihood_7_test():
    onhdp = model((1, 2, 2), D=47, W=42)
    doc = Document([0, 41, 3], [2, 1, 2])
    assert_almost_equal(-9.235786743,
        onhdp.w_likelihood(
        doc,
        np.array([
            [0.125, 0.125, 0.125, 0.125, 0.125, 0.25, 0.125],
            [0.0625, 0.0625, 0.0625, 0.0625, 0.25, 0.25, 0.25],
            [0.03125, 0.25, 0.125, 0.03125, 0.03125, 0.5, 0.03125],
            [0.125, 0.125, 0.5, 0.0625, 0.03125, 0.03125, 0.125],
            [0.0625, 0.0625, 0.03125, 0.75, 0.03125, 0.03125, 0.03125],
        ]),
        np.log(np.array([
            [0.125, 0.5, 0.25],
            [0.03125, 0.75, 0.125],
            [0.75, 0.125, 0.0625],
            [0.625, 0.03125, 0.25],
            [0.375, 0.125, 0.03125],
            [0.0625, 0.03125, 0.875],
            [0.625, 0.125, 0.125],
        ])),
        [0, 1, 2, 3, 4, 5, 6]))

def w_likelihood_8_test():
    onhdp = model((1, 2, 2), D=47, W=42)
    doc = Document([0, 41, 3], [2, 1, 2])
    assert_almost_equal(-3.755323728,
        onhdp.w_likelihood(
        doc,
        np.array([
            [0.125, 0.125, 0.125, 0.125, 0.125, 0.25, 0.125],
            [0.0625, 0.0625, 0.0625, 0.0625, 0.25, 0.25, 0.25],
            [0.03125, 0.25, 0.125, 0.03125, 0.03125, 0.5, 0.03125],
            [0.125, 0.125, 0.5, 0.0625, 0.03125, 0.03125, 0.125],
            [0.0625, 0.0625, 0.03125, 0.75, 0.03125, 0.03125, 0.03125],
        ]),
        np.log(np.array([
            [0.125, 0.5, 0.25],
            [0.03125, 0.75, 0.125],
            [0.75, 0.125, 0.0625],
            [0.625, 0.03125, 0.25],
            [0.375, 0.125, 0.03125],
            [0.0625, 0.03125, 0.875],
            [0.625, 0.125, 0.125],
        ])),
        [0, 1, 3, 4]))
