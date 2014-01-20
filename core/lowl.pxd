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
    ctypedef unsigned int lowl_count

    const int LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS
    const int LOWLERR_BADMALLOC
    const int LOWLERR_BADINPUT
    const int LOWLERR_INDEXOUTOFRANGE


cdef extern from "lowl_sketch.h":
    ctypedef struct bloomfilter:
        pass

    int  bloomfilter_init(bloomfilter* f, size_t size, unsigned int k)
    void bloomfilter_insert(bloomfilter* f, const char *x, size_t n)
    bint bloomfilter_query(bloomfilter* f, const char *x, size_t n)
    void bloomfilter_print(bloomfilter* f)
    void bloomfilter_write(bloomfilter* f, FILE* fp)
    int bloomfilter_read(bloomfilter* f, FILE* fp)
    void bloomfilter_destroy(bloomfilter* f)

    ctypedef struct cmsketch:
        pass

    int cmsketch_init(cmsketch* cm, size_t w, size_t d)
    int cmsketch_add(cmsketch* cm, const char *x, size_t n, lowl_count delta)
    lowl_count cmsketch_query(cmsketch* cm, const char *x, size_t n)
    void cmsketch_print(cmsketch* cm)
    void cmsketch_write(cmsketch* cm, FILE* fp)
    int cmsketch_read(cmsketch* cm, FILE* fp)
    void cmsketch_clear(cmsketch* cm)
    void cmsketch_destroy(cmsketch* cm)


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
    int reservoirsampler_get(reservoirsampler* rs, size_t idx, lowl_key *x)
    lowl_key* reservoirsampler_sample(reservoirsampler* rs)
