from libc.stdio cimport FILE
from libc.stddef cimport size_t

cdef extern from "stdio.h":
    FILE *fopen(const char *filename, const char *mode)
    int fclose(FILE *f)

cdef extern from "lowl_types.h":
    ctypedef unsigned long lowl_key
    ctypedef unsigned int lowl_hashoutput
    ctypedef unsigned int lowl_count

cdef extern from "lowl_sketch.h":
    ctypedef struct bloomfilter:
        pass

    void bloomfilter_init(bloomfilter* f, size_t size, unsigned int k)
    void bloomfilter_insertKey(bloomfilter* f, lowl_key k)
    int  bloomfilter_queryKey(bloomfilter* f, lowl_key k)
    void bloomfilter_print(bloomfilter* f)
    void bloomfilter_write(bloomfilter* f, FILE* fp)
    void bloomfilter_read(bloomfilter* f, FILE* fp)
    void bloomfilter_destroy(bloomfilter* f)
