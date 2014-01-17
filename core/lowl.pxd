from libc.stdio cimport FILE
from libc.stddef cimport size_t

cdef extern from "stdio.h":
    FILE *fopen(const char *filename, const char *mode)
    int fclose(FILE *f)

cdef extern from "stdlib.h":
    void srandom(unsigned int seed)

cdef extern from "lowl_types.h":
    ctypedef unsigned int lowl_hashoutput
    ctypedef unsigned int lowl_key

cdef extern from "lowl_sketch.h":
    ctypedef struct bloomfilter:
        pass

    int  bloomfilter_init(bloomfilter* f, size_t size, unsigned int k)
    void bloomfilter_insert(bloomfilter* f, const char *x, size_t x_len)
    bint bloomfilter_query(bloomfilter* f, const char *x, size_t x_len)
    void bloomfilter_print(bloomfilter* f)
    void bloomfilter_write(bloomfilter* f, FILE* fp)
    int bloomfilter_read(bloomfilter* f, FILE* fp)
    void bloomfilter_destroy(bloomfilter* f)

cdef extern from "lowl_sample.h":
    ctypedef struct reservoirsampler:
        pass

    int  reservoirsampler_init(reservoirsampler* rs, size_t capacity)
    bint reservoirsampler_insert(reservoirsampler* rs, lowl_key x, size_t *idx, bint *ejected, lowl_key *ejected_key)
    void reservoirsampler_print(reservoirsampler* rs)
    void reservoirsampler_write(reservoirsampler* rs, FILE* fp)
    int reservoirsampler_read(reservoirsampler* rs, FILE* fp)
    size_t reservoirsampler_capacity(reservoirsampler* rs)
    size_t reservoirsampler_occupied(reservoirsampler* rs)
    void reservoirsampler_destroy(reservoirsampler* rs)
    int reservoirsampler_sample(reservoirsampler* rs, size_t *idx)
    int reservoirsampler_get(reservoirsampler* rs, size_t idx, lowl_key *x)
    lowl_key* reservoirsampler_get_all(reservoirsampler* rs)
