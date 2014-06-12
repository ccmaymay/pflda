cimport lowl
from libc.stddef cimport size_t
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cPickle import load, dump


def _raises(f, exception_type):
    try:
        f
    except exception_type:
        return False
    else:
        return True


def _chisq(expected, observed):
    """
    Compute chi-squared statistic for the given sequences of expected
    and observed counts.

    >>> from pylowl import _chisq
    >>> _chisq([3], [3]) == 0.0
    True
    >>> _chisq([1], [3]) == 4.0
    True
    >>> _chisq([3], [1]) == 4.0 / 3.0
    True
    >>> _chisq([1, 3, 5], [1, 3, 5]) == 0.0
    True
    >>> _chisq([5, 3, 1], [1, 3, 5]) == 16.0 / 5.0 + 16.0
    True
    >>> _chisq([1, 3, 5], [5, 3, 1]) == 16.0 / 5.0 + 16.0
    True
    >>> _chisq([2, 4, 6], [1, 3, 5]) == 0.25 + 0.5 + 1.0 / 6.0
    True

    This space intentionally left blank.
    """

    total = 0.0
    for (exp, obs) in zip(expected, observed):
        diff = exp - obs
        total += diff * diff / float(exp)
    return total


cdef int _check_err(int ret) except -1:
    """
    Raise exception according to standard integral lowl error code.
    """

    if ret == lowl.LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS:
        return 0
    elif ret == lowl.LOWLERR_BADMALLOC:
        raise MemoryError()
    elif ret == lowl.LOWLERR_BADINPUT:
        raise ValueError()
    elif ret == lowl.LOWLERR_INDEXOUTOFRANGE:
        raise IndexError()
    else:
        raise ValueError("Unknown return code.")


cpdef srandom(unsigned int seed):
    """
    Seed the PRNG used in lowl.

    Check that calling srandom with different seeds yields different
    PRNGs while calling srandom the the same seed yields the same PRNG.
    >>> from pylowl import srandom, ReservoirSampler
    >>> srandom(0)
    >>> rs1 = ReservoirSampler()
    >>> ret = rs1.init(1)
    >>> for i in xrange(1000000):
    ...     (inserted, idx, ejected, ejected_key) = rs1.insert(i)
    >>> k1 = rs1.get(0)
    >>> srandom(42)
    >>> rs2 = ReservoirSampler()
    >>> ret = rs2.init(1)
    >>> for i in xrange(1000000):
    ...     (inserted, idx, ejected, ejected_key) = rs2.insert(i)
    >>> k2 = rs2.get(0)
    >>> srandom(0)
    >>> rs3 = ReservoirSampler()
    >>> ret = rs3.init(1)
    >>> for i in xrange(1000000):
    ...     (inserted, idx, ejected, ejected_key) = rs3.insert(i)
    >>> k3 = rs3.get(0)
    >>> k1 == k2
    False
    >>> k1 == k3
    True

    I'm claustrophobic.
    """
    lowl.srandom(seed)


cdef class BloomFilter:
    """
    Bloom filter for string (const char *) elements.

    Test basic bloom filter behavior.
    >>> from pylowl import BloomFilter
    >>> bf = BloomFilter()
    >>> ret = bf.init(4, 8)
    >>> bf.insert("hello, world", 12)
    >>> bf.insert("hello world", 11)
    >>> bf.insert("hello, waldorf", 14)
    >>> bf.query("hello, world", 12)
    True
    >>> bf.query("hello world", 11)
    True
    >>> bf.query("hello, waldo", 12)
    False
    >>> bf.query("hello, waldorf", 14)
    True
    >>> bf.query("hello, waldorf!", 15)
    False
    >>> bf.query("hello, waldorf!", 14)
    True

    Test serialization and deserialization.
    >>> from tempfile import mkstemp
    >>> import os
    >>> (fid, filename) = mkstemp()
    >>> os.close(fid)
    >>> ret = bf.write(filename)
    >>> bf_fromfile = BloomFilter()
    >>> ret = bf_fromfile.read(filename)
    >>> bf_fromfile.query("hello, world", 12)
    True
    >>> bf_fromfile.query("hello world", 11)
    True
    >>> bf_fromfile.query("hello, waldo", 12)
    False
    >>> bf_fromfile.query("hello, waldorf", 14)
    True
    >>> bf_fromfile.query("hello, waldorf!", 15)
    False
    >>> bf_fromfile.query("hello, waldorf!", 14)
    True
    >>> bf_fromfile.query("foobar", 6)
    False
    >>> bf_fromfile.insert("foobar!", 6)
    >>> bf_fromfile.query("foobar", 6)
    True
    >>> os.remove(filename)

    Test serialization and deserialization errors.
    >>> from tempfile import mkdtemp
    >>> from pylowl import _raises
    >>> _raises(lambda: bf.write("/this/path/should/not/be/writable"), IOError)
    True
    >>> dirname = mkdtemp()
    >>> _raises(lambda: bf.write(dirname), IOError)
    True
    >>> os.rmdir(dirname)
    >>> bf = BloomFilter()
    >>> _raises(lambda: bf.read("/this/path/should/not/exist"), IOError)
    True
    >>> (fid, filename) = mkstemp()
    >>> os.close(fid)
    >>> bf = BloomFilter()
    >>> _raises(lambda: bf.read(filename), IOError)
    True
    >>> os.remove(filename)

    Check boundary cases on parameters.
    >>> bf = BloomFilter()
    >>> ret = bf.init(1, 1)
    >>> bf.insert("hello, world", 12)
    >>> bf.query("hello, world", 12)
    True

    Check that an uninitialized filter does not cause an abort when
    it is deallocated.  (If this fails it will crash the test runner!)
    >>> bf_noinit = BloomFilter()

    That newline was magical.
    """

    def __cinit__(self):
        self._bf = NULL

    cpdef int init(self, size_t size, unsigned int k) except -1:
        cdef int ret

        self._bf = <lowl.bloomfilter *>PyMem_Malloc(sizeof(lowl.bloomfilter))
        if self._bf is NULL:
            raise MemoryError()

        ret = _check_err(lowl.bloomfilter_init(self._bf, size, k))
        if ret != 0:
            return -1

        return 0

    cpdef insert(self, const char* x, size_t n):
        lowl.bloomfilter_insert(self._bf, x, n)

    cpdef bint query(self, const char* x, size_t n):
        return lowl.bloomfilter_query(self._bf, x, n)

    cpdef prnt(self):
        lowl.bloomfilter_print(self._bf)

    cpdef int read(self, const char* filename) except -1:
        cdef lowl.FILE* f
        cdef int ret

        if self._bf is not NULL:
            lowl.bloomfilter_destroy(self._bf)
            PyMem_Free(self._bf)
        self._bf = <lowl.bloomfilter *>PyMem_Malloc(sizeof(lowl.bloomfilter))
        if self._bf is NULL:
            raise MemoryError()

        f = lowl.fopen(filename, 'rb')
        if f is NULL:
            raise IOError("Failed to open file.")
        ret = _check_err(lowl.bloomfilter_read(self._bf, f))
        if ret != 0:
            return -1
        ret = lowl.fclose(f)
        if ret != 0:
            raise IOError("Failed to close file.")

        return 0

    cpdef int write(self, const char* filename) except -1:
        cdef lowl.FILE* f
        cdef int ret

        f = lowl.fopen(filename, 'wb')
        if f is NULL:
            raise IOError("Failed to open file.")
        lowl.bloomfilter_write(self._bf, f)
        ret = lowl.fclose(f)
        if ret != 0:
            raise IOError("Failed to close file.")

        return 0

    def __dealloc__(self):
        if self._bf is not NULL:
            lowl.bloomfilter_destroy(self._bf)
            PyMem_Free(self._bf)


cdef class CountMinSketch:
    """
    CM sketch for string (const char *) elements.

    Test basic CM sketch behavior.
    >>> from pylowl import CountMinSketch
    >>> cm = CountMinSketch()
    >>> ret = cm.init(4, 8)
    >>> cm.add("hello, world", 12, 1)
    >>> cm.add("hello world", 11, 1)
    >>> cm.add("hello, world", 12, 4)
    >>> cm.add("hello, waldorf", 14, 42)
    >>> cm.query("hello, world", 12) == 5
    True
    >>> cm.query("hello world", 11) == 1
    True
    >>> cm.query("hello, waldo", 12) == 0
    True
    >>> cm.query("hello, waldorf", 14) == 42
    True
    >>> cm.query("hello, waldorf!", 15) == 0
    True
    >>> cm.query("hello, waldorf!", 14) == 42
    True

    Test serialization and deserialization.
    >>> from tempfile import mkstemp
    >>> import os
    >>> (fid, filename) = mkstemp()
    >>> os.close(fid)
    >>> ret = cm.write(filename)
    >>> cm_fromfile = CountMinSketch()
    >>> ret = cm_fromfile.read(filename)
    >>> cm_fromfile.query("hello, world", 12) == 5
    True
    >>> cm_fromfile.query("hello world", 11) == 1
    True
    >>> cm_fromfile.query("hello, waldo", 12) == 0
    True
    >>> cm_fromfile.query("hello, waldorf", 14) == 42
    True
    >>> cm_fromfile.query("hello, waldorf!", 15) == 0
    True
    >>> cm_fromfile.query("hello, waldorf!", 14) == 42
    True
    >>> cm_fromfile.query("foobar", 6) == 0
    True
    >>> cm_fromfile.add("foobar!", 6, 7)
    >>> cm_fromfile.query("foobar", 6) == 7
    True
    >>> os.remove(filename)

    Test serialization and deserialization errors.
    >>> from tempfile import mkdtemp
    >>> from pylowl import _raises
    >>> _raises(lambda: cm.write("/this/path/should/not/be/writable"), IOError)
    True
    >>> dirname = mkdtemp()
    >>> _raises(lambda: cm.write(dirname), IOError)
    True
    >>> os.rmdir(dirname)
    >>> cm = CountMinSketch()
    >>> _raises(lambda: cm.read("/this/path/should/not/exist"), IOError)
    True
    >>> (fid, filename) = mkstemp()
    >>> os.close(fid)
    >>> cm = CountMinSketch()
    >>> _raises(lambda: cm.read(filename), IOError)
    True
    >>> os.remove(filename)

    Test clear.
    >>> cm = CountMinSketch()
    >>> ret = cm.init(4, 8)
    >>> cm.add("hello, world", 12, 1)
    >>> cm.add("hello world", 11, 1)
    >>> cm.add("hello, world", 12, 4)
    >>> cm.add("hello, waldorf", 14, 42)
    >>> cm.clear()
    >>> cm.query("hello, world", 12) == 0
    True
    >>> cm.query("hello world", 11) == 0
    True
    >>> cm.query("hello, waldo", 12) == 0
    True
    >>> cm.query("hello, waldorf", 14) == 0
    True
    >>> cm.query("hello, waldorf!", 15) == 0
    True
    >>> cm.query("hello, waldorf!", 14) == 0
    True

    Check boundary cases on parameters.
    >>> cm = CountMinSketch()
    >>> ret = cm.init(1, 1)
    >>> cm.add("hello, world", 12, 42)
    >>> cm.query("hello, world", 12) == 42
    True

    Check that an uninitialized sketch does not cause an abort when
    it is deallocated.  (If this fails it will crash the test runner!)
    >>> cm_noinit = CountMinSketch()

    It's important to take (line)breaks.
    """

    def __cinit__(self):
        self._cm = NULL

    cpdef int init(self, size_t w, size_t d) except -1:
        cdef int ret

        self._cm = <lowl.cmsketch *>PyMem_Malloc(sizeof(lowl.cmsketch))
        if self._cm is NULL:
            raise MemoryError()

        ret = _check_err(lowl.cmsketch_init(self._cm, w, d))
        if ret != 0:
            return -1

        return 0

    cpdef add(self, const char* x, size_t n, lowl.lowl_count delta):
        lowl.cmsketch_add(self._cm, x, n, delta)

    cpdef lowl.lowl_count query(self, const char* x, size_t n):
        return lowl.cmsketch_query(self._cm, x, n)

    cpdef prnt(self):
        lowl.cmsketch_print(self._cm)

    cpdef clear(self):
        lowl.cmsketch_clear(self._cm)

    cpdef int read(self, const char* filename) except -1:
        cdef lowl.FILE* f
        cdef int ret

        if self._cm is not NULL:
            lowl.cmsketch_destroy(self._cm)
            PyMem_Free(self._cm)
        self._cm = <lowl.cmsketch *>PyMem_Malloc(sizeof(lowl.cmsketch))
        if self._cm is NULL:
            raise MemoryError()

        f = lowl.fopen(filename, 'rb')
        if f is NULL:
            raise IOError("Failed to open file.")
        ret = _check_err(lowl.cmsketch_read(self._cm, f))
        if ret != 0:
            return -1
        ret = lowl.fclose(f)
        if ret != 0:
            raise IOError("Failed to close file.")

        return 0

    cpdef int write(self, const char* filename) except -1:
        cdef lowl.FILE* f
        cdef int ret

        f = lowl.fopen(filename, 'wb')
        if f is NULL:
            raise IOError("Failed to open file.")
        lowl.cmsketch_write(self._cm, f)
        ret = lowl.fclose(f)
        if ret != 0:
            raise IOError("Failed to close file.")

        return 0

    def __dealloc__(self):
        if self._cm is not NULL:
            lowl.cmsketch_destroy(self._cm)
            PyMem_Free(self._cm)


cdef class ReservoirSampler:
    """
    Reservoir sampler for lowl_key (integral) elements.

    Test basic reservoir sampler behavior.
    >>> from pylowl import ReservoirSampler
    >>> rs = ReservoirSampler()
    >>> ret = rs.init(4)
    >>> (rs.capacity(), rs.occupied()) == (4, 0)
    True
    >>> rs.insert(42)[:3] == (True, 0, False)
    True
    >>> rs.insert(47)[:3] == (True, 1, False)
    True
    >>> rs.insert(3)[:3] == (True, 2, False)
    True
    >>> (rs.capacity(), rs.occupied()) == (4, 3)
    True
    >>> rs.insert(52)[:3] == (True, 3, False)
    True
    >>> (rs.capacity(), rs.occupied()) == (4, 4)
    True
    >>> (inserted, idx, ejected, ejected_key) = rs.insert(7)
    >>> inserted == ejected
    True
    >>> (not inserted) or (idx in range(4))
    True
    >>> initial_keys = set([42, 47, 3, 52])
    >>> (not ejected) or (ejected_key in initial_keys)
    True
    >>> (rs.capacity(), rs.occupied()) == (4, 4)
    True

    Assert basic expectations over a long stream:  Eventually we will
    insert and eject something, and in particular, we will eventually
    eject the four initial reservoir elements.  Also assert that
    insertion occurs if and only if ejection occurs, and that our sample
    is a subset of all items seen in the stream.
    >>> n = 10000
    >>> inserted_any = False
    >>> ejected_any = False
    >>> inserted_xor_ejected = False
    >>> ejected_keys_in_sample = True
    >>> sample = [rs.get(i) for i in range(4)]
    >>> for i in xrange(n):
    ...     (inserted, idx, ejected, ejected_key) = rs.insert(i)
    ...     inserted_any |= inserted
    ...     ejected_any |= ejected
    ...     ejected_keys_in_sample &= (not ejected) or (ejected_key in sample)
    ...     inserted_xor_ejected |= (inserted ^ ejected)
    ...     sample = [rs.get(j) for j in range(4)]
    >>> inserted_any
    True
    >>> ejected_any
    True
    >>> inserted_xor_ejected
    False
    >>> ejected_keys_in_sample
    True

    Test serialization and deserialization.
    >>> from tempfile import mkstemp
    >>> import os
    >>> (fid, filename) = mkstemp()
    >>> os.close(fid)
    >>> ret = rs.write(filename)
    >>> rs_fromfile = ReservoirSampler()
    >>> ret = rs_fromfile.read(filename)
    >>> (rs_fromfile.capacity(), rs_fromfile.occupied()) == (4, 4)
    True
    >>> sample_fromfile = [rs_fromfile.get(i) for i in range(4)]
    >>> sample == sample_fromfile
    True
    >>> for i in range(4):
    ...     (inserted, idx, ejected, ejected_key) = rs_fromfile.insert(i + n)
    >>> sample_fromfile = [rs_fromfile.get(i) for i in range(4)]
    >>> set([i + n for i in range(4)]).isdisjoint(set(sample_fromfile))
    True
    >>> os.remove(filename)

    Test serialization and deserialization errors.
    >>> from tempfile import mkdtemp
    >>> from pylowl import _raises
    >>> _raises(lambda: rs.write("/this/path/should/not/be/writable"), IOError)
    True
    >>> dirname = mkdtemp()
    >>> _raises(lambda: rs.write(dirname), IOError)
    True
    >>> os.rmdir(dirname)
    >>> rs = ReservoirSampler()
    >>> _raises(lambda: rs.read("/this/path/should/not/exist"), IOError)
    True
    >>> (fid, filename) = mkstemp()
    >>> os.close(fid)
    >>> rs = ReservoirSampler()
    >>> _raises(lambda: rs.read(filename), IOError)
    True
    >>> os.remove(filename)

    Check that sample returns a contiguous memoryview (read: efficient
    array-like wrapper) on the occupied fraction of the sample.
    >>> rs = ReservoirSampler()
    >>> ret = rs.init(4)
    >>> (inserted, idx, ejected, ejected_key) = rs.insert(42)
    >>> (inserted, idx, ejected, ejected_key) = rs.insert(47)
    >>> (inserted, idx, ejected, ejected_key) = rs.insert(3)
    >>> xx = rs.sample()
    >>> xx.strides == (xx.itemsize,)
    True
    >>> xx.shape == (3,)
    True
    >>> xx.is_c_contig()
    True
    >>> (xx[0], xx[1], xx[2]) == (42, 47, 3)
    True

    Check that exceptions are raised properly when get fails.
    >>> x = rs.get(2)
    >>> raised = False
    >>> try:
    ...     rs.get(3) # >= occupied, < capacity
    ... except IndexError:
    ...     raised = True
    >>> raised
    True
    >>> raised = False
    >>> try:
    ...     rs.get(4) # >= occupied, >= capacity
    ... except IndexError:
    ...     raised = True
    >>> raised
    True

    Show that if our reservoir keys are the numbers 1 through 8 and
    the reservoir size is 2, then every 2-set of distinct numbers
    (8 choose 2 of these) has an equal probability of being the
    reservoir.  (Run n different experiments and show that the
    distribution of 2-sets is uniform.)
    >>> n = 10000
    >>> expected = [n / 28.0] * 28
    >>> observed = dict()
    >>> for i in range(8):
    ...     for j in range(i):
    ...         observed[(j, i)] = 0
    >>> for i in xrange(n):
    ...     rs = ReservoirSampler()
    ...     ret = rs.init(2)
    ...     for j in range(8):
    ...         quad = rs.insert(j)
    ...     sample = (rs.get(0), rs.get(1))
    ...     observed[(min(sample), max(sample))] += 1
    >>> _chisq(expected, observed.values()) < 36.74122 # df = 27, alpha = 0.1
    True

    Check boundary cases on parameters.
    >>> rs = ReservoirSampler()
    >>> ret = rs.init(1)
    >>> (inserted, idx, ejected, ejected_key) = rs.insert(42)
    >>> (inserted, idx, ejected, ejected_key) = rs.insert(12)
    >>> rs.get(0) in (42, 12)
    True

    Check that an uninitialized reservoir does not cause an abort when
    it is deallocated.  (If this fails it will crash the test runner!)
    >>> rs_noinit = ReservoirSampler()

    Newlines keep the compiler happy.
    """

    def __cinit__(self):
        self._rs = NULL

    cpdef int init(self, size_t capacity) except -1:
        cdef int ret

        self._rs = <lowl.reservoirsampler *>PyMem_Malloc(sizeof(lowl.reservoirsampler))
        if self._rs is NULL:
            raise MemoryError()

        ret = _check_err(lowl.reservoirsampler_init(self._rs, capacity))
        if ret != 0:
            return -1

        return 0

    cdef bint _insert(self, lowl.lowl_key k, size_t* idx, bint* ejected, lowl.lowl_key* ejected_key):
        cdef bint inserted
        inserted = lowl.reservoirsampler_insert(self._rs, k, idx, ejected, ejected_key)
        return inserted

    def insert(self, lowl.lowl_key k):
        cdef bint inserted
        cdef size_t idx
        cdef bint ejected
        cdef lowl.lowl_key ejected_key
        inserted = self._insert(k, &idx, &ejected, &ejected_key)
        return (inserted, idx, ejected, ejected_key)

    cpdef int read(self, const char* filename) except -1:
        cdef lowl.FILE* f
        cdef int ret

        if self._rs is not NULL:
            lowl.reservoirsampler_destroy(self._rs)
            PyMem_Free(self._rs)
        self._rs = <lowl.reservoirsampler *>PyMem_Malloc(sizeof(lowl.reservoirsampler))
        if self._rs is NULL:
            raise MemoryError()

        f = lowl.fopen(filename, 'rb')
        if f is NULL:
            raise IOError("Failed to open file.")
        ret = _check_err(lowl.reservoirsampler_read(self._rs, f))
        if ret != 0:
            return -1
        ret = lowl.fclose(f)
        if ret != 0:
            raise IOError("Failed to close file.")

        return 0

    cpdef int write(self, const char* filename) except -1:
        cdef lowl.FILE* f
        cdef int ret

        f = lowl.fopen(filename, 'wb')
        if f is NULL:
            raise IOError("Failed to open file.")
        lowl.reservoirsampler_write(self._rs, f)
        ret = lowl.fclose(f)
        if ret != 0:
            raise IOError("Failed to close file.")

        return 0

    cpdef size_t capacity(self):
        return lowl.reservoirsampler_capacity(self._rs)

    cpdef size_t occupied(self):
        return lowl.reservoirsampler_occupied(self._rs)

    cpdef prnt(self):
        lowl.reservoirsampler_print(self._rs)

    cpdef lowl.lowl_key get(self, idx) except *:
        cdef lowl.lowl_key k
        _check_err(lowl.reservoirsampler_get(self._rs, idx, &k))
        return k

    cdef lowl.lowl_key* _sample(self):
        cdef lowl.lowl_key* xx
        xx = lowl.reservoirsampler_sample(self._rs)
        return xx

    cpdef lowl.lowl_key [::1] sample(self):
        cdef lowl.lowl_key [::1] xx_view
        cdef size_t occupied
        occupied = self.occupied()
        xx_view = <lowl.lowl_key[:occupied]> self._sample()
        return xx_view

    def __dealloc__(self):
        if self._rs is not NULL:
            lowl.reservoirsampler_destroy(self._rs)
            PyMem_Free(self._rs)


class ValuedReservoirSampler(object):
    """
    Reservoir sampler for arbitrary Python objects as elements.

    Test basic reservoir sampler behavior.
    >>> from pylowl import ValuedReservoirSampler
    >>> rs = ValuedReservoirSampler(4)
    >>> (rs.capacity(), rs.occupied()) == (4, 0)
    True
    >>> rs.sample() == []
    True
    >>> rs.insert(42)[:3] == (True, 0, False)
    True
    >>> rs.insert("Foobar")[:3] == (True, 1, False)
    True
    >>> rs.insert(47)[:3] == (True, 2, False)
    True
    >>> (rs.capacity(), rs.occupied()) == (4, 3)
    True
    >>> rs.sample() == [42, "Foobar", 47]
    True
    >>> rs.insert(dict(foo="bar"))[:3] == (True, 3, False)
    True
    >>> (rs.capacity(), rs.occupied()) == (4, 4)
    True
    >>> sample = [x for x in rs.sample()]
    >>> sample == [42, "Foobar", 47, dict(foo="bar")]
    True
    >>> (inserted, idx, ejected, ejected_val) = rs.insert(set([7, 37]))
    >>> inserted == ejected
    True
    >>> (not inserted) or (idx in range(4))
    True
    >>> initial_vals = [42, "Foobar", 47, dict(foo="bar")]
    >>> (not ejected) or (ejected_val in initial_vals)
    True
    >>> (rs.capacity(), rs.occupied()) == (4, 4)
    True
    >>> if inserted:
    ...     sample[idx] = set([7, 37])
    >>> sample == rs.sample()
    True

    Assert basic expectations over a long stream:  Eventually we will
    insert and eject something, and in particular, we will eventually
    eject the four initial reservoir elements.  Also assert that
    insertion occurs if and only if ejection occurs, and that our sample
    is a subset of all items seen in the stream.
    >>> n = 10000
    >>> inserted_any = False
    >>> ejected_any = False
    >>> inserted_xor_ejected = False
    >>> ejected_vals_in_sample = True
    >>> for i in xrange(n):
    ...     sample = [x for x in rs.sample()]
    ...     (inserted, idx, ejected, ejected_val) = rs.insert(i)
    ...     inserted_any |= inserted
    ...     ejected_any |= ejected
    ...     ejected_vals_in_sample &= (not ejected) or (ejected_val in sample)
    ...     inserted_xor_ejected |= (inserted ^ ejected)
    >>> inserted_any
    True
    >>> ejected_any
    True
    >>> inserted_xor_ejected
    False
    >>> ejected_vals_in_sample
    True

    Test serialization and deserialization.
    >>> from tempfile import mkstemp
    >>> import os
    >>> (fid, filename) = mkstemp()
    >>> os.close(fid)
    >>> (fid, values_filename) = mkstemp()
    >>> os.close(fid)
    >>> rs.write(filename, values_filename)
    >>> rs_fromfile = ValuedReservoirSampler.read(filename, values_filename)
    >>> (rs_fromfile.capacity(), rs_fromfile.occupied()) == (4, 4)
    True
    >>> rs.sample() == rs_fromfile.sample()
    True
    >>> inserted_any = False
    >>> for i in range(4):
    ...     (inserted, idx, ejected, ejected_val) = rs_fromfile.insert(i + n)
    ...     inserted_any |= inserted
    >>> inserted_any
    False
    >>> os.remove(filename)
    >>> os.remove(values_filename)

    Test serialization and deserialization errors.
    >>> from tempfile import mkdtemp
    >>> from pylowl import _raises
    >>> (fid, filename) = mkstemp()
    >>> os.close(fid)
    >>> _raises(lambda: rs.write(filename, "/this/path/should/not/be/writable"), IOError)
    True
    >>> _raises(lambda: rs.write("/this/path/should/not/be/writable", filename), IOError)
    True
    >>> dirname = mkdtemp()
    >>> _raises(lambda: rs.write(filename, dirname), IOError)
    True
    >>> _raises(lambda: rs.write(dirname, filename), IOError)
    True
    >>> os.rmdir(dirname)
    >>> (fid, values_filename) = mkstemp()
    >>> os.close(fid)
    >>> rs_fromfile.write(filename, values_filename)
    >>> _raises(lambda: ValuedReservoirSampler.read(filename, "/this/path/should/not/exist"), IOError)
    True
    >>> _raises(lambda: ValuedReservoirSampler.read("/this/path/should/not/exist", values_filename), IOError)
    True
    >>> (fid, empty_filename) = mkstemp()
    >>> os.close(fid)
    >>> _raises(lambda: ValuedReservoirSampler.read(filename, empty_filename), IOError)
    True
    >>> _raises(lambda: ValuedReservoirSampler.read(empty_filename, values_filename), IOError)
    True
    >>> os.remove(empty_filename)
    >>> os.remove(filename)
    >>> os.remove(values_filename)

    Check that exceptions are raised properly when get fails.
    >>> rs = ValuedReservoirSampler(4)
    >>> (inserted, idx, ejected, ejected_val) = rs.insert(42)
    >>> (inserted, idx, ejected, ejected_val) = rs.insert("Foobar")
    >>> (inserted, idx, ejected, ejected_val) = rs.insert(47)
    >>> x = rs.get(2)
    >>> raised = False
    >>> try:
    ...     rs.get(3) # >= occupied, < capacity
    ... except IndexError:
    ...     raised = True
    >>> raised
    True
    >>> raised = False
    >>> try:
    ...     rs.get(4) # >= occupied, >= capacity
    ... except IndexError:
    ...     raised = True
    >>> raised
    True

    Show that if our reservoir vals are the numbers 1 through 8 and
    the reservoir size is 2, then every 2-set of distinct numbers
    (8 choose 2 of these) has an equal probability of being the
    reservoir.  (Run n different experiments and show that the
    distribution of 2-sets is uniform.)
    >>> n = 10000
    >>> expected = [n / 28.0] * 28
    >>> observed = dict()
    >>> for i in range(8):
    ...     for j in range(i):
    ...         observed[(j, i)] = 0
    >>> for i in xrange(n):
    ...     rs = ValuedReservoirSampler(2)
    ...     for j in range(8):
    ...         quad = rs.insert(j)
    ...     observed[(min(rs.sample()), max(rs.sample()))] += 1
    >>> _chisq(expected, observed.values()) < 36.74122 # df = 27, alpha = 0.1
    True

    Check boundary cases on parameters.
    >>> rs = ValuedReservoirSampler(1)
    >>> (inserted, idx, ejected, ejected_val) = rs.insert(42)
    >>> (inserted, idx, ejected, ejected_val) = rs.insert(12)
    >>> rs.get(0) in (42, 12)
    True
    >>> rs = ValuedReservoirSampler(0)

    This newline is valued transitively.
    """
    def __init__(self, size_t capacity):
        self.rs = ReservoirSampler()
        if capacity > 0:
            self.rs.init(capacity)
        self.values = [None] * capacity
        self.unused_key = 0

    def insert(self, object v, object preprocess=None):
        (inserted, idx, ejected, ejected_key) = self.rs.insert(self.unused_key)
        if inserted:
            if ejected:
                ejected_val = self.values[idx]
                self.unused_key = ejected_key
            else:
                ejected_val = None
                self.unused_key += 1
            if preprocess is None:
                self.values[idx] = v
            else:
                self.values[idx] = preprocess(v)
        else:
            ejected_val = None
        return (inserted, idx, ejected, ejected_val)

    @classmethod
    def read(self, const char* filename, const char* values_filename):
        vrs = ValuedReservoirSampler(0)
        vrs._read(filename, values_filename)
        return vrs

    def _read(self, const char* filename, const char* values_filename):
        self.rs.read(filename)
        self.values = [None] * self.rs.capacity()
        with open(values_filename, 'r') as f:
            self.values = load(f)
        if self.rs.occupied() < self.rs.capacity():
            self.unused_key = self.rs.occupied()
        else:
            unused_keys = set(range(self.rs.capacity() + 1))
            for i in xrange(self.rs.capacity()):
                unused_keys.remove(self.rs.get(i))
            self.unused_key = unused_keys.pop()

    def write(self, const char* filename, const char* values_filename):
        self.rs.write(filename)
        with open(values_filename, 'w') as f:
            dump(self.values, f)

    def capacity(self):
        return self.rs.capacity()

    def occupied(self):
        return self.rs.occupied()

    def prnt(self):
        self.rs.prnt()

    def sample(self):
        return self.values[:self.occupied()]

    def get(self, size_t idx):
        if idx >= self.occupied():
            raise IndexError()
        return self.values[idx]
