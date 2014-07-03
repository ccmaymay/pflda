from numpy.testing import assert_almost_equal
from tempfile import mkstemp, mkdtemp
import os

from pylowl.core import *


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


def bloomfilter_test():
    bf = BloomFilter()
    ret = bf.init(4, 8)
    bf.insert("hello, world", 12)
    bf.insert("hello world", 11)
    bf.insert("hello, waldorf", 14)

    assert bf.query("hello, world", 12)
    assert bf.query("hello world", 11)
    assert not bf.query("hello, waldo", 12)
    assert bf.query("hello, waldorf", 14)
    assert not bf.query("hello, waldorf!", 15)
    assert bf.query("hello, waldorf!", 14)

def bloomfilter_serialization_test():
    bf = BloomFilter()
    ret = bf.init(8, 8)
    bf.insert("hello, world", 12)
    bf.insert("hello world", 11)
    bf.insert("hello, waldorf", 14)

    (fid, filename) = mkstemp()
    os.close(fid)
    ret = bf.write(filename)
    bf_fromfile = BloomFilter()
    ret = bf_fromfile.read(filename)
    assert bf_fromfile.query("hello, world", 12)
    assert bf_fromfile.query("hello world", 11)
    assert not bf_fromfile.query("hello, waldo", 12)
    assert bf_fromfile.query("hello, waldorf", 14)
    assert not bf_fromfile.query("hello, waldorf!", 15)
    assert bf_fromfile.query("hello, waldorf!", 14)
    assert not bf_fromfile.query("foobar", 6)

    bf_fromfile.insert("foobar!", 6)
    assert bf_fromfile.query("foobar", 6)
    os.remove(filename)

def bloomfilter_serialization_errors_test():
    bf = BloomFilter()
    ret = bf.init(4, 8)
    bf.insert("hello, world", 12)
    bf.insert("hello world", 11)
    bf.insert("hello, waldorf", 14)

    assert raises(lambda: bf.write("/this/path/should/not/be/writable"), IOError)

    dirname = mkdtemp()
    assert raises(lambda: bf.write(dirname), IOError)
    os.rmdir(dirname)

    bf = BloomFilter()
    assert raises(lambda: bf.read("/this/path/should/not/exist"), IOError)

    (fid, filename) = mkstemp()
    os.close(fid)
    bf = BloomFilter()
    assert raises(lambda: bf.read(filename), IOError)
    os.remove(filename)

def bloomfilter_param_bndy_cases_test():
    bf = BloomFilter()
    ret = bf.init(1, 1)
    bf.insert("hello, world", 12)
    assert bf.query("hello, world", 12)

def bloomfilter_uninitialized_deallocation_test():
    def f():
        bf_noinit = BloomFilter()
    f()


def cmsketch_test():
    cm = CountMinSketch()
    ret = cm.init(4, 8)
    cm.add("hello, world", 12, 1)
    cm.add("hello world", 11, 1)
    cm.add("hello, world", 12, 4)
    cm.add("hello, waldorf", 14, 42)

    assert cm.query("hello, world", 12) == 5
    assert cm.query("hello world", 11) == 1
    assert cm.query("hello, waldo", 12) == 0
    assert cm.query("hello, waldorf", 14) == 42
    assert cm.query("hello, waldorf!", 15) == 0
    assert cm.query("hello, waldorf!", 14) == 42

def cmsketch_serialization_test():
    cm = CountMinSketch()
    ret = cm.init(4, 8)
    cm.add("hello, world", 12, 1)
    cm.add("hello world", 11, 1)
    cm.add("hello, world", 12, 4)
    cm.add("hello, waldorf", 14, 42)

    (fid, filename) = mkstemp()
    os.close(fid)
    ret = cm.write(filename)
    cm_fromfile = CountMinSketch()
    ret = cm_fromfile.read(filename)
    assert cm_fromfile.query("hello, world", 12) == 5
    assert cm_fromfile.query("hello world", 11) == 1
    assert cm_fromfile.query("hello, waldo", 12) == 0
    assert cm_fromfile.query("hello, waldorf", 14) == 42
    assert cm_fromfile.query("hello, waldorf!", 15) == 0
    assert cm_fromfile.query("hello, waldorf!", 14) == 42
    assert cm_fromfile.query("foobar", 6) == 0
    cm_fromfile.add("foobar!", 6, 7)
    assert cm_fromfile.query("foobar", 6) == 7
    os.remove(filename)

def cmsketch_serialization_errors_test():
    assert False
#    cm = CountMinSketch()
#    ret = cm.init(4, 8)
#    cm.add("hello, world", 12, 1)
#    cm.add("hello world", 11, 1)
#    cm.add("hello, world", 12, 4)
#    cm.add("hello, waldorf", 14, 42)
#
#    assert raises(lambda: cm.write("/this/path/should/not/be/writable"), IOError)
#
#    dirname = mkdtemp()
#    assert raises(lambda: cm.write(dirname), IOError)
#    os.rmdir(dirname)
#
#    cm = CountMinSketch()
#    assert raises(lambda: cm.read("/this/path/should/not/exist"), IOError)
#
#    (fid, filename) = mkstemp()
#    os.close(fid)
#    cm = CountMinSketch()
#    assert raises(lambda: cm.read(filename), IOError)
#    os.remove(filename)

def cmsketch_clear_test():
    cm = CountMinSketch()
    ret = cm.init(4, 8)
    cm.add("hello, world", 12, 1)
    cm.add("hello world", 11, 1)
    cm.add("hello, world", 12, 4)
    cm.add("hello, waldorf", 14, 42)
    cm.clear()
    assert cm.query("hello, world", 12) == 0
    assert cm.query("hello world", 11) == 0
    assert cm.query("hello, waldo", 12) == 0
    assert cm.query("hello, waldorf", 14) == 0
    assert cm.query("hello, waldorf!", 15) == 0
    assert cm.query("hello, waldorf!", 14) == 0

def cmsketch_param_bndy_cases_test():
    cm = CountMinSketch()
    ret = cm.init(1, 1)
    cm.add("hello, world", 12, 42)
    assert cm.query("hello, world", 12) == 42

def cmsketch_uninitialized_deallocation_test():
    def f():
        cm_noinit = CountMinSketch()
    f()
