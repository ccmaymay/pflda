cimport lowl
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cPickle import load, dump


def _chisq(expected, observed):
    """
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


cpdef srandom(unsigned int seed):
    """
    >>> from pylowl import srandom
    >>> srandom(0)
    >>> srandom(42)

    I'm claustrophobic.
    """
    lowl.srandom(seed)


cdef class BloomFilter:
    """
    >>> from pylowl import BloomFilter
    >>> bf = BloomFilter()
    >>> bf.init(4, 8)
    >>> bf.insert("hello, world")
    >>> bf.insert("hello world")
    >>> bf.insert("hello, waldorf")
    >>> bf.query("hello, world")
    True
    >>> bf.query("hello world")
    True
    >>> bf.query("hello, waldo")
    False
    >>> bf.query("hello, waldorf")
    True
    >>> bf_noinit = BloomFilter()

    That newline was magical.
    """

    cdef lowl.bloomfilter* _bf

    def __cinit__(self):
        self._bf = NULL

    def init(self, size, k):
        self._bf = <lowl.bloomfilter *>PyMem_Malloc(sizeof(lowl.bloomfilter))
        if self._bf is NULL:
            raise MemoryError()
        lowl.bloomfilter_init(self._bf, size, k)
        # TODO error code

    def insert(self, x):
        lowl.bloomfilter_insert(self._bf, x, len(x))

    def query(self, x):
        return lowl.bloomfilter_query(self._bf, x, len(x))

    def prnt(self):
        lowl.bloomfilter_print(self._bf)

    def read(self, filename):
        f = lowl.fopen(filename, 'rb')
        lowl.bloomfilter_read(self._bf, f)
        # TODO error code
        lowl.fclose(f)

    def write(self, filename):
        f = lowl.fopen(filename, 'wb')
        lowl.bloomfilter_write(self._bf, f)
        lowl.fclose(f)

    def __dealloc__(self):
        if self._bf is not NULL:
            lowl.bloomfilter_destroy(self._bf)
            PyMem_Free(self._bf)


cdef class ReservoirSampler:
    """
    >>> from pylowl import ReservoirSampler
    >>> rs = ReservoirSampler()
    >>> rs.init(4)
    >>> rs.insert(42)[:3]
    (True, 0L, False)
    >>> rs.insert(47)[:3]
    (True, 1L, False)
    >>> rs.insert(3)[:3]
    (True, 2L, False)
    >>> rs.insert(52)[:3]
    (True, 3L, False)
    >>> quad = rs.insert(7)
    >>> quad[0] == quad[2] # inserted iff ejected
    True
    >>> (not quad[0]) or (quad[1] in range(4))
    True
    >>> (not quad[2]) or (quad[3] in (42, 47, 3, 52))
    True
    >>> inserted = False
    >>> ejected = False
    >>> ejected_vals = set()
    >>> for i in range(10000):
    ...     quad = rs.insert(i)
    ...     inserted |= quad[0]
    ...     ejected_vals.add(quad[3])
    ...     ejected |= quad[2]
    >>> set([42, 47, 3, 52]).issubset(ejected_vals)
    True
    >>> inserted
    True
    >>> ejected
    True
    >>> n = 10000
    >>> expected = [n / 28.0] * 28
    >>> observed = dict()
    >>> for i in range(8):
    ...     for j in range(i):
    ...         observed[(j, i)] = 0
    >>> for i in range(n):
    ...     rs = ReservoirSampler()
    ...     rs.init(2)
    ...     for j in range(8):
    ...         quad = rs.insert(j)
    ...     sample = (rs.get(0), rs.get(1))
    ...     observed[(min(sample), max(sample))] += 1
    >>> _chisq(expected, observed.values()) < 36.74122 # df = 27, alpha = 0.1
    True
    >>> rs_noinit = ReservoirSampler()

    Newlines keep the compiler happy.
    """

    cdef lowl.reservoirsampler* _rs

    def __cinit__(self):
        self._rs = NULL

    cpdef init(self, lowl.size_t capacity):
        self._rs = <lowl.reservoirsampler *>PyMem_Malloc(sizeof(lowl.reservoirsampler))
        if self._rs is NULL:
            raise MemoryError()
        lowl.reservoirsampler_init(self._rs, capacity)
        # TODO error code

    cdef bint c_insert(self, lowl.lowl_key k, lowl.size_t* idx, bint* ejected, lowl.lowl_key* ejected_key):
        cdef bint inserted
        inserted = lowl.reservoirsampler_insert(self._rs, k, idx, ejected, ejected_key)
        return inserted

    def insert(self, lowl.lowl_key k):
        cdef bint inserted
        cdef lowl.size_t idx
        cdef bint ejected
        cdef lowl.lowl_key ejected_key
        inserted = self.c_insert(k, &idx, &ejected, &ejected_key)
        return (inserted, idx, ejected, ejected_key)

    cpdef read(self, const char* filename):
        cdef lowl.FILE* f
        f = lowl.fopen(filename, 'rb')
        lowl.reservoirsampler_read(self._rs, f)
        # TODO error code
        lowl.fclose(f)

    cpdef write(self, const char* filename):
        cdef lowl.FILE* f
        f = lowl.fopen(filename, 'wb')
        lowl.reservoirsampler_write(self._rs, f)
        lowl.fclose(f)

    cpdef lowl.size_t capacity(self):
        return lowl.reservoirsampler_capacity(self._rs)

    cpdef lowl.size_t occupied(self):
        return lowl.reservoirsampler_occupied(self._rs)

    cpdef prnt(self):
        lowl.reservoirsampler_print(self._rs)

    cpdef lowl.size_t sample(self):
        cdef lowl.size_t idx
        lowl.reservoirsampler_sample(self._rs, &idx)
        # TODO error code
        return idx

    cpdef lowl.lowl_key get(self, idx):
        cdef int ret
        cdef lowl.lowl_key k
        ret = lowl.reservoirsampler_get(self._rs, idx, &k)
        # TODO bounds/ret check
        return k

    def __dealloc__(self):
        if self._rs is not NULL:
            lowl.reservoirsampler_destroy(self._rs)
            PyMem_Free(self._rs)


class ValuedReservoirSampler(object):
    """
    >>> from pylowl import ValuedReservoirSampler
    >>> rs = ValuedReservoirSampler(4)

    This newline is valued transitively.
    """
    def __init__(self, lowl.size_t capacity):
        self.rs = ReservoirSampler()
        self.rs.init(capacity)
        self.values = [None] * capacity
        self.unused_key = 0

    def insert(self, object v):
        (inserted, idx, ejected, ejected_key) = self.rs.insert(self.unused_key)
        if inserted:
            if ejected:
                ejected_val = self.values[idx]
                self.unused_key = ejected_key
            else:
                ejected_val = None
                self.unused_key += 1
            self.values[idx] = v
        else:
            ejected_val = None
        return (inserted, idx, ejected, ejected_val)

    def read(self, const char* filename, const char* values_filename):
        self.rs.read(filename)
        self.values = [None] * self.rs.capacity()
        with open(values_filename, 'r') as f:
            self.values = load(f)
        if self.rs.occupied() < self.rs.capacity():
            self.unused_key = self.rs.occupied()
        else:
            unused_keys = set(range(self.rs.capacity() + 1))
            for i in range(self.rs.capacity()):
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

    def get(self, lowl.size_t idx):
        # TODO check
        return self.values[idx]

    def sample(self):
        # TODO check
        return self.values[self.rs.sample()]
