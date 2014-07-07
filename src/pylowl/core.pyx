cimport lowl
from libc.stddef cimport size_t
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cPickle import load, dump


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
    """
    lowl.srandom(seed)


cdef class BloomFilter:
    """
    Bloom filter for string (const char *) elements.

    Test basic bloom filter behavior.
    """

    def __cinit__(self):
        self._bf = NULL

    cpdef int init(self, size_t size, unsigned int k) except -1:
        cdef int ret

        self._bf = <lowl.bloomfilter *>PyMem_Malloc(sizeof(lowl.bloomfilter))
        if self._bf is NULL:
            raise MemoryError()
        lowl.bloomfilter_preinit(self._bf)

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
        lowl.bloomfilter_preinit(self._bf)

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
    """

    def __cinit__(self):
        self._cm = NULL

    cpdef int init(self, size_t w, size_t d) except -1:
        cdef int ret

        self._cm = <lowl.cmsketch *>PyMem_Malloc(sizeof(lowl.cmsketch))
        if self._cm is NULL:
            raise MemoryError()
        lowl.cmsketch_preinit(self._cm)

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
        lowl.cmsketch_preinit(self._cm)

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
    """

    def __cinit__(self):
        self._rs = NULL

    cpdef int init(self, size_t capacity) except -1:
        cdef int ret

        self._rs = <lowl.reservoirsampler *>PyMem_Malloc(sizeof(lowl.reservoirsampler))
        if self._rs is NULL:
            raise MemoryError()
        lowl.reservoirsampler_preinit(self._rs)

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
        lowl.reservoirsampler_preinit(self._rs)

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

    cpdef lowl.lowl_key get(self, idx) except? 64321:
        cdef lowl.lowl_key k
        cdef int ret
        ret = _check_err(lowl.reservoirsampler_get(self._rs, idx, &k))
        if ret != 0:
            return 64321
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
