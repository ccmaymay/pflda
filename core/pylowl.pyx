cimport lowl
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free


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

    def cPrint(self):
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


cdef class ValuedReservoirSampler:
    cdef lowl.reservoirsampler* _rs
    cdef void** _values
    cdef lowl.lowl_key _unused_key

    def __cinit__(self):
        self._rs = <lowl.reservoirsampler *>PyMem_Malloc(sizeof(lowl.reservoirsampler))
        if self._rs is NULL:
            raise MemoryError()
        self._values = NULL
        self._unused_key = 0

    cpdef init(self, lowl.size_t capacity):
        lowl.reservoirsampler_init(self._rs, capacity)
        # TODO error code
        self._values = <void **>PyMem_Malloc(sizeof(void*) * capacity)
        self._unused_key = 0

    cdef bint c_insert(self, void* v, void** ejected_val):
        cdef lowl.size_t idx
        cdef bint ejected
        cdef lowl.lowl_key ejected_key
        cdef bint inserted
        inserted = lowl.reservoirsampler_insert(self._rs, self._unused_key, &idx, &ejected, &ejected_key)
        if inserted:
            ejected_val[0] = self._values[idx]
            self._values[idx] = v
            if ejected:
                self._unused_key = ejected_key
            else:
                self._unused_key += 1
                ejected_val[0] = NULL
        else:
            ejected_val[0] = NULL
        cdef int i
        for i in range(self.occupied()):
            print(<object> (self._values[i]))
        return inserted

    def insert(self, v):
        cdef void* ejected_val
        cdef bint inserted
        inserted = self.c_insert(<void *> v, &ejected_val)
        ejected_obj = None
        if ejected_val is not NULL:
            ejected_obj = <object> ejected_val
        return (inserted, ejected_obj)

    def read(self, filename, values_filename):
        self.c_read(filename)
        with open(values_filename, 'r') as f:
            i = 0
            for line in f:
                stripped_line = line.rstrip()
                self._values[i] = <void *> stripped_line
                i += 1

    cdef void c_read(self, const char* filename):
        cdef lowl.FILE* f
        f = lowl.fopen(filename, 'rb')
        lowl.reservoirsampler_read(self._rs, f)
        # TODO error code
        lowl.fclose(f)

    def write(self, filename, values_filename):
        self.c_write(filename)
        with open(values_filename, 'w') as f:
            for i in range(self.capacity()):
                f.write(str(<object> self._values[i]) + '\n')

    cdef void c_write(self, const char* filename):
        cdef lowl.FILE* f
        f = lowl.fopen(filename, 'wb')
        lowl.reservoirsampler_write(self._rs, f)
        lowl.fclose(f)

    cdef lowl.size_t capacity(self):
        return lowl.reservoirsampler_capacity(self._rs)

    cdef lowl.size_t occupied(self):
        return lowl.reservoirsampler_occupied(self._rs)

    cdef void cPrint(self):
        lowl.reservoirsampler_print(self._rs)

    cdef lowl.size_t sample(self):
        cdef lowl.size_t idx
        lowl.reservoirsampler_sample(self._rs, &idx)
        # TODO error code
        return idx

    cdef void* get(self, idx):
        return self._values[idx]

    def __dealloc__(self):
        if self._rs is not NULL:
            lowl.reservoirsampler_destroy(self._rs)
            PyMem_Free(self._rs)
        if self._values is not NULL:
            PyMem_Free(self._values)
