cimport lowl
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.cobject cimport PyCObject_AsVoidPtr, PyCObject_FromVoidPtr


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
        return bool(lowl.bloomfilter_query(self._bf, x, len(x)))

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


cdef class ReservoirSampler:
    cdef lowl.reservoirsampler* _rs
    cdef void** values

    def __cinit__(self):
        self._rs = <lowl.reservoirsampler *>PyMem_Malloc(sizeof(lowl.reservoirsampler))
        if self._rs is NULL:
            raise MemoryError()

    def init(self, capacity):
        lowl.reservoirsampler_init(self._rs, capacity)
        self.values = <void **>PyMem_Malloc(sizeof(void*) * capacity)
        if self.values is NULL:
            raise MemoryError()
        # TODO error code

    def insert(self, k, v):
        cdef lowl.size_t idx
        cdef lowl.lowl_key ejected
        inserted = bool(lowl.reservoirsampler_insert(self._rs, k, &idx, &ejected))
        if inserted:
            ejected_value = self.values[idx]
            self.values[idx] = PyCObject_AsVoidPtr(v)
            if ejected_value is NULL:
                return None
            else:
                return PyCObject_FromVoidPtr(ejected_value, NULL)
        else:
            return None

    def read(self, filename, values_filename):
        f = lowl.fopen(filename, 'rb')
        lowl.reservoirsampler_read(self._rs, f)
        # TODO error code
        lowl.fclose(f)

        if self.values is not NULL:
            PyMem_Free(self.values)
        self.values = <void **>PyMem_Malloc(sizeof(void*) * self.capacity())
        if self.values is NULL:
            raise MemoryError()
        with open(values_filename, 'r') as vf:
            i = 0
            for line in vf:
                if i < self.occupied():
                    self.values[i] = PyCObject_AsVoidPtr(line)
                i += 1

    def write(self, filename, values_filename):
        f = lowl.fopen(filename, 'wb')
        lowl.reservoirsampler_write(self._rs, f)
        lowl.fclose(f)

        with open(values_filename, 'w') as vf:
            for i in range(self.occupied()):
                vf.write(PyCObject_FromVoidPtr(self.values[i], NULL) + '\n')

    def capacity(self):
        return lowl.reservoirsampler_capacity(self._rs)

    def occupied(self):
        return lowl.reservoirsampler_occupied(self._rs)

    def cPrint(self):
        lowl.reservoirsampler_print(self._rs)

    def __dealloc__(self):
        if self._rs is not NULL:
            lowl.reservoirsampler_destroy(self._rs)
            PyMem_Free(self._rs)
        if self.values is not NULL:
            PyMem_Free(self.values)
