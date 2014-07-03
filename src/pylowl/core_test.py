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


def reservoirsampler_test():
    rs = ReservoirSampler()
    ret = rs.init(4)
    assert (rs.capacity(), rs.occupied()) == (4, 0)
    assert rs.insert(42)[:3] == (True, 0, False)
    assert rs.insert(47)[:3] == (True, 1, False)
    assert rs.insert(3)[:3] == (True, 2, False)
    assert (rs.capacity(), rs.occupied()) == (4, 3)
    assert rs.insert(52)[:3] == (True, 3, False)
    assert (rs.capacity(), rs.occupied()) == (4, 4)
    (inserted, idx, ejected, ejected_key) = rs.insert(7)
    assert inserted == ejected
    assert (not inserted) or (idx in range(4))
    initial_keys = set([42, 47, 3, 52])
    assert (not ejected) or (ejected_key in initial_keys)
    assert (rs.capacity(), rs.occupied()) == (4, 4)

def reservoirsampler_positive_probability_event_test():
    '''
    Assert basic expectations over a long stream:  Eventually we will
    insert and eject something, and in particular, we will eventually
    eject the four initial reservoir elements.  Also assert that
    insertion occurs if and only if ejection occurs, and that our sample
    is a subset of all items seen in the stream.
    '''
    rs = ReservoirSampler()
    ret = rs.init(4)
    rs.insert(42)
    rs.insert(47)
    rs.insert(3)
    rs.insert(52)
    rs.insert(7)

    n = 10000
    inserted_any = False
    ejected_any = False
    inserted_xor_ejected = False
    ejected_keys_in_sample = True
    sample = [rs.get(i) for i in range(4)]
    for i in xrange(n):
        (inserted, idx, ejected, ejected_key) = rs.insert(i)
        inserted_any |= inserted
        ejected_any |= ejected
        ejected_keys_in_sample &= (not ejected) or (ejected_key in sample)
        inserted_xor_ejected |= (inserted ^ ejected)
        sample = [rs.get(j) for j in range(4)]
    assert inserted_any
    assert ejected_any
    assert not inserted_xor_ejected
    assert ejected_keys_in_sample

def reservoirsampler_serialization_test():
    rs = ReservoirSampler()
    ret = rs.init(4)
    rs.insert(42)
    rs.insert(47)
    rs.insert(3)
    rs.insert(52)
    rs.insert(7)
    n = 10000
    for i in xrange(n):
        (inserted, idx, ejected, ejected_key) = rs.insert(i)
    sample = [rs.get(j) for j in range(4)]

    (fid, filename) = mkstemp()
    os.close(fid)
    ret = rs.write(filename)
    rs_fromfile = ReservoirSampler()
    ret = rs_fromfile.read(filename)
    assert (rs_fromfile.capacity(), rs_fromfile.occupied()) == (4, 4)
    sample_fromfile = [rs_fromfile.get(i) for i in range(4)]
    assert sample == sample_fromfile
    for i in range(4):
        (inserted, idx, ejected, ejected_key) = rs_fromfile.insert(i + n)
    sample_fromfile = [rs_fromfile.get(i) for i in range(4)]
    assert set([i + n for i in range(4)]).isdisjoint(set(sample_fromfile))
    os.remove(filename)

def reservoirsampler_serialization_errors_test():
    rs = ReservoirSampler()
    ret = rs.init(4)
    rs.insert(42)
    rs.insert(47)
    rs.insert(3)
    rs.insert(52)
    rs.insert(7)

    assert raises(lambda: rs.write("/this/path/should/not/be/writable"), IOError)

    dirname = mkdtemp()
    assert raises(lambda: rs.write(dirname), IOError)
    os.rmdir(dirname)

    rs = ReservoirSampler()
    assert raises(lambda: rs.read("/this/path/should/not/exist"), IOError)

    (fid, filename) = mkstemp()
    os.close(fid)
    rs = ReservoirSampler()
    assert raises(lambda: rs.read(filename), IOError)
    os.remove(filename)

def reservoirsampler_c_contig_test():
    rs = ReservoirSampler()
    ret = rs.init(4)
    (inserted, idx, ejected, ejected_key) = rs.insert(42)
    (inserted, idx, ejected, ejected_key) = rs.insert(47)
    (inserted, idx, ejected, ejected_key) = rs.insert(3)

    xx = rs.sample()
    assert xx.strides == (xx.itemsize,)
    assert xx.shape == (3,)
    assert xx.is_c_contig()
    assert (xx[0], xx[1], xx[2]) == (42, 47, 3)

def reservoirsampler_get_errors_test():
    rs = ReservoirSampler()
    ret = rs.init(4)
    (inserted, idx, ejected, ejected_key) = rs.insert(42)
    (inserted, idx, ejected, ejected_key) = rs.insert(47)
    (inserted, idx, ejected, ejected_key) = rs.insert(3)

    x = rs.get(2)
    assert raises(lambda: rs.get(3), IndexError)
    assert raises(lambda: rs.get(4), IndexError)

def reservoirsampler_unif_expectation_test():
    '''
    Show that if our reservoir keys are the numbers 1 through 8 and
    the reservoir size is 2, then every 2-set of distinct numbers
    (8 choose 2 of these) has an equal probability of being the
    reservoir.  (Run n different experiments and show that the
    distribution of 2-sets is uniform.)
    '''
    n = 10000
    expected = [n / 28.0] * 28
    observed = dict()
    for i in range(8):
        for j in range(i):
            observed[(j, i)] = 0
    for i in xrange(n):
        rs = ReservoirSampler()
        ret = rs.init(2)
        for j in range(8):
            quad = rs.insert(j)
        sample = (rs.get(0), rs.get(1))
        observed[(min(sample), max(sample))] += 1
    assert chisq(expected, observed.values()) < 36.74122 # df = 27, alpha = 0.1

def reservoirsampler_param_bndy_cases_test():
    rs = ReservoirSampler()
    ret = rs.init(1)
    (inserted, idx, ejected, ejected_key) = rs.insert(42)
    (inserted, idx, ejected, ejected_key) = rs.insert(12)
    assert rs.get(0) in (42, 12)

def reservoirsampler_uninitialized_deallocation_test():
    def f():
        rs_noinit = ReservoirSampler()
    f()


def valuedreservoirsampler_test():
    rs = ValuedReservoirSampler(4)
    assert (rs.capacity(), rs.occupied()) == (4, 0)
    assert rs.sample() == []
    assert rs.insert(42)[:3] == (True, 0, False)
    assert rs.insert("Foobar")[:3] == (True, 1, False)
    assert rs.insert(47)[:3] == (True, 2, False)
    assert (rs.capacity(), rs.occupied()) == (4, 3)
    assert rs.sample() == [42, "Foobar", 47]
    assert rs.insert(dict(foo="bar"))[:3] == (True, 3, False)
    assert (rs.capacity(), rs.occupied()) == (4, 4)
    sample = [x for x in rs.sample()]
    assert sample == [42, "Foobar", 47, dict(foo="bar")]
    (inserted, idx, ejected, ejected_val) = rs.insert(set([7, 37]))
    assert inserted == ejected
    assert (not inserted) or (idx in range(4))
    initial_vals = [42, "Foobar", 47, dict(foo="bar")]
    assert (not ejected) or (ejected_val in initial_vals)
    assert (rs.capacity(), rs.occupied()) == (4, 4)
    if inserted:
        sample[idx] = set([7, 37])
    assert sample == rs.sample()

def valuedreservoirsampler_positive_probability_event_test():
    '''
    Assert basic expectations over a long stream:  Eventually we will
    insert and eject something, and in particular, we will eventually
    eject the four initial reservoir elements.  Also assert that
    insertion occurs if and only if ejection occurs, and that our sample
    is a subset of all items seen in the stream.
    '''
    rs = ValuedReservoirSampler(4)
    rs.insert(42)[:3] == (True, 0, False)
    rs.insert("Foobar")[:3] == (True, 1, False)
    rs.insert(47)[:3] == (True, 2, False)
    rs.insert(dict(foo="bar"))[:3] == (True, 3, False)
    rs.insert(set([7, 37]))
    n = 10000

    inserted_any = False
    ejected_any = False
    inserted_xor_ejected = False
    ejected_vals_in_sample = True
    for i in xrange(n):
        sample = [x for x in rs.sample()]
        (inserted, idx, ejected, ejected_val) = rs.insert(i)
        inserted_any |= inserted
        ejected_any |= ejected
        ejected_vals_in_sample &= (not ejected) or (ejected_val in sample)
        inserted_xor_ejected |= (inserted ^ ejected)
    assert inserted_any
    assert ejected_any
    assert not inserted_xor_ejected
    assert ejected_vals_in_sample

def valuedreservoirsampler_serialization_test():
    rs = ValuedReservoirSampler(4)
    rs.insert(42)[:3] == (True, 0, False)
    rs.insert("Foobar")[:3] == (True, 1, False)
    rs.insert(47)[:3] == (True, 2, False)
    rs.insert(dict(foo="bar"))[:3] == (True, 3, False)
    rs.insert(set([7, 37]))
    n = 10000
    for i in xrange(n):
        rs.insert(i)

    (fid, filename) = mkstemp()
    os.close(fid)
    (fid, values_filename) = mkstemp()
    os.close(fid)
    rs.write(filename, values_filename)
    rs_fromfile = ValuedReservoirSampler.read(filename, values_filename)
    assert (rs_fromfile.capacity(), rs_fromfile.occupied()) == (4, 4)
    assert rs.sample() == rs_fromfile.sample()
    inserted_any = False
    for i in range(4):
        (inserted, idx, ejected, ejected_val) = rs_fromfile.insert(i + n)
        inserted_any |= inserted
    assert not inserted_any
    os.remove(filename)
    os.remove(values_filename)

def valuedreservoirsampler_serialization_errors_test():
    rs = ValuedReservoirSampler(4)
    rs.insert(42)[:3] == (True, 0, False)
    rs.insert("Foobar")[:3] == (True, 1, False)
    rs.insert(47)[:3] == (True, 2, False)
    rs.insert(dict(foo="bar"))[:3] == (True, 3, False)
    rs.insert(set([7, 37]))
    n = 10000
    for i in xrange(n):
        rs.insert(i)

    (fid, filename) = mkstemp()
    os.close(fid)
    (fid, values_filename) = mkstemp()
    os.close(fid)
    rs.write(filename, values_filename)
    rs_fromfile = ValuedReservoirSampler.read(filename, values_filename)

    (fid, filename) = mkstemp()
    os.close(fid)
    assert raises(lambda: rs.write(filename, "/this/path/should/not/be/writable"), IOError)
    assert raises(lambda: rs.write("/this/path/should/not/be/writable", filename), IOError)
    dirname = mkdtemp()
    assert raises(lambda: rs.write(filename, dirname), IOError)
    assert raises(lambda: rs.write(dirname, filename), IOError)
    os.rmdir(dirname)
    (fid, values_filename) = mkstemp()
    os.close(fid)
    rs_fromfile.write(filename, values_filename)
    assert raises(lambda: ValuedReservoirSampler.read(filename, "/this/path/should/not/exist"), IOError)
    assert raises(lambda: ValuedReservoirSampler.read("/this/path/should/not/exist", values_filename), IOError)
    (fid, empty_filename) = mkstemp()
    os.close(fid)
    assert raises(lambda: ValuedReservoirSampler.read(filename, empty_filename), IOError)
    assert raises(lambda: ValuedReservoirSampler.read(empty_filename, values_filename), IOError)
    os.remove(empty_filename)
    os.remove(filename)
    os.remove(values_filename)

def valuedreservoirsampler_get_errors_test():
    rs = ValuedReservoirSampler(4)
    (inserted, idx, ejected, ejected_val) = rs.insert(42)
    (inserted, idx, ejected, ejected_val) = rs.insert("Foobar")
    (inserted, idx, ejected, ejected_val) = rs.insert(47)
    x = rs.get(2)
    assert raises(lambda: rs.get(3), IndexError)
    assert raises(lambda: rs.get(4), IndexError)

def valuedreservoirsampler_unif_expectation_test():
    '''
    Show that if our reservoir vals are the numbers 1 through 8 and
    the reservoir size is 2, then every 2-set of distinct numbers
    (8 choose 2 of these) has an equal probability of being the
    reservoir.  (Run n different experiments and show that the
    distribution of 2-sets is uniform.)
    '''
    n = 10000
    expected = [n / 28.0] * 28
    observed = dict()
    for i in range(8):
        for j in range(i):
            observed[(j, i)] = 0
    for i in xrange(n):
        rs = ValuedReservoirSampler(2)
        for j in range(8):
            quad = rs.insert(j)
        observed[(min(rs.sample()), max(rs.sample()))] += 1
    assert chisq(expected, observed.values()) < 36.74122 # df = 27, alpha = 0.1

def valuedreservoirsampler_param_bndy_cases_test():
    rs = ValuedReservoirSampler(1)
    (inserted, idx, ejected, ejected_val) = rs.insert(42)
    (inserted, idx, ejected, ejected_val) = rs.insert(12)
    assert rs.get(0) in (42, 12)

    assert not raises(lambda: ValuedReservoirSampler(0), Exception)
