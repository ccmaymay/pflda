from libc.stdio cimport FILE
from libc.stddef cimport size_t

cdef extern from "stdio.h":
    FILE *fopen(const char *filename, const char *mode)
    int fclose(FILE *f)

cdef extern from "lowl_types.h":
    ctypedef unsigned int lowl_hashoutput

cdef extern from "lowl_sketch.h":
    ctypedef struct bloomfilter:
        pass

    void bloomfilter_init(bloomfilter* f, size_t size, unsigned int k)
    void bloomfilter_insert(bloomfilter* f, const char *x, size_t x_len)
    int  bloomfilter_query(bloomfilter* f, const char *x, size_t x_len)
    void bloomfilter_print(bloomfilter* f)
    void bloomfilter_write(bloomfilter* f, FILE* fp)
    void bloomfilter_read(bloomfilter* f, FILE* fp)
    void bloomfilter_destroy(bloomfilter* f)
