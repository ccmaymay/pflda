cimport lowl
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class BloomFilter:
    cdef lowl.lowl_bloomfilter* _bf

    def __cinit__(self):
        self._bf = <lowl.lowl_bloomfilter *>PyMem_Malloc(sizeof(lowl.lowl_bloomfilter))
        if self._bf is NULL:
            raise MemoryError()

    def init(self, size, k):
        lowl.lowl_bloomfilter_init(self._bf, size, k)

    def insert(self, k):
        lowl.lowl_bloomfilter_insertKey(self._bf, k)

    def query(self, k):
        lowl.lowl_bloomfilter_queryKey(self._bf, k)

    def cPrint(self):
        lowl.lowl_bloomfilter_print(self._bf)

    #def read(self, fp):
    #    lowl.lowl_bloomfilter_read(self._bf, fp)

    #def write(self, fp):
    #    lowl.lowl_bloomfilter_write(self._bf, fp)

    def __dealloc__(self):
        if self._bf is not NULL:
            lowl.lowl_bloomfilter_destroy(self._bf)
            PyMem_Free(self._bf)
