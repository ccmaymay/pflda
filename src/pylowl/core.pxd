from libc.stddef cimport size_t

cdef extern from "lowl_types.h":
    ctypedef unsigned int lowl_hashoutput
    ctypedef unsigned int lowl_key
    ctypedef unsigned int lowl_count

cdef int _check_err(int ret) except -1
cpdef srandom(unsigned int seed)

cdef extern from "lowl_sketch.h":
    ctypedef struct bloomfilter:
        pass
    ctypedef struct cmsketch:
        pass

cdef extern from "lowl_sample.h":
    ctypedef struct reservoirsampler:
        pass

cdef class BloomFilter:
    cdef bloomfilter* _bf
    cpdef int init(self, size_t size, unsigned int k) except -1
    cpdef insert(self, const char* x, size_t n)
    cpdef bint query(self, const char* x, size_t n)
    cpdef prnt(self)
    cpdef int read(self, const char* filename) except -1
    cpdef int write(self, const char* filename) except -1

cdef class CountMinSketch:
    cdef cmsketch* _cm
    cpdef int init(self, size_t w, size_t d) except -1
    cpdef add(self, const char* x, size_t n, lowl_count delta)
    cpdef lowl_count query(self, const char* x, size_t n)
    cpdef prnt(self)
    cpdef clear(self)
    cpdef int read(self, const char* filename) except -1
    cpdef int write(self, const char* filename) except -1

cdef class ReservoirSampler:
    cdef reservoirsampler* _rs
    cpdef int init(self, size_t capacity) except -1
    cdef bint _insert(self, lowl_key k, size_t* idx, bint* ejected, lowl_key* ejected_key)
    #def insert(self, lowl_key k)
    cpdef int read(self, const char* filename) except -1
    cpdef int write(self, const char* filename) except -1
    cpdef size_t capacity(self)
    cpdef size_t occupied(self)
    cpdef prnt(self)
    cpdef lowl_key get(self, idx) except? 64321
    cdef lowl_key* _sample(self)
    cpdef lowl_key [::1] sample(self)
