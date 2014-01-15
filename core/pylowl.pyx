cimport lowl
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cPickle import load, dump


def srandom(seed):
    lowl.srandom(seed)


cdef class BloomFilter:
    cdef lowl.bloomfilter* _bf

    def __cinit__(self):
        self._bf = <lowl.bloomfilter *>PyMem_Malloc(sizeof(lowl.bloomfilter))
        if self._bf is NULL:
            raise MemoryError()

    def init(self, size, k):
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
    cdef lowl.reservoirsampler* _rs

    def __cinit__(self):
        self._rs = <lowl.reservoirsampler *>PyMem_Malloc(sizeof(lowl.reservoirsampler))
        if self._rs is NULL:
            raise MemoryError()

    cpdef init(self, lowl.size_t capacity):
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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
