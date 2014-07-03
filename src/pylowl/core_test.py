from pylowl.core import *
from numpy.testing import assert_almost_equal


def raises(f, ex_type):
    '''
    Return true iff f, when called, raises an exception of type ex_type
    '''

    try:
        f()
    except ex_type:
        return True
    except:
        return False
    else:
        return False


def chisq(expected, observed):
    """
    Compute chi-squared statistic for the given sequences of expected
    and observed counts.
    """
    total = 0.0
    for (exp, obs) in zip(expected, observed):
        diff = exp - obs
        total += diff * diff / float(exp)
    return total


def chisq_1_test():
    assert_almost_equal(chisq([3], [3]), 0.0)

def chisq_2_test():
    assert_almost_equal(chisq([1], [3]), 4.0)

def chisq_3_test():
    assert_almost_equal(chisq([3], [1]), 4.0 / 3.0)

def chisq_4_test():
    assert_almost_equal(chisq([1, 3, 5], [1, 3, 5]), 0.0)

def chisq_5_test():
    assert_almost_equal(chisq([5, 3, 1], [1, 3, 5]), 16.0 / 5.0 + 16.0)

def chisq_6_test():
    assert_almost_equal(chisq([1, 3, 5], [5, 3, 1]), 16.0 / 5.0 + 16.0)

def chisq_7_test():
    assert_almost_equal(chisq([2, 4, 6], [1, 3, 5]), 0.25 + 0.5 + 1.0 / 6.0)

def srandom_test():
    srandom(0)
    rs1 = ReservoirSampler()
    ret = rs1.init(1)
    for i in xrange(1000000):
        (inserted, idx, ejected, ejected_key) = rs1.insert(i)
    k1 = rs1.get(0)
    srandom(42)
    rs2 = ReservoirSampler()
    ret = rs2.init(1)
    for i in xrange(1000000):
        (inserted, idx, ejected, ejected_key) = rs2.insert(i)
    k2 = rs2.get(0)
    srandom(0)
    rs3 = ReservoirSampler()
    ret = rs3.init(1)
    for i in xrange(1000000):
        (inserted, idx, ejected, ejected_key) = rs3.insert(i)
    k3 = rs3.get(0)

    assert k1 != k2
    assert k1 == k3
