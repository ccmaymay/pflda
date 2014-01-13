cdef extern from "lowl_types.h":
    ctypedef unsigned long lowl_key
    ctypedef unsigned int lowl_hashoutput
    ctypedef unsigned int lowl_count

cdef extern from "lowl_bloom.h":
    ctypedef struct lowl_bloomfilter:
        pass

    void lowl_bloomfilter_init(lowl_bloomfilter* f, int size, int k)
    void lowl_bloomfilter_insertKey(lowl_bloomfilter* f, lowl_key k)
    int  lowl_bloomfilter_queryKey(lowl_bloomfilter* f, lowl_key k)
    void lowl_bloomfilter_print(lowl_bloomfilter* f)
    void lowl_bloomfilter_destroy(lowl_bloomfilter* f)

    #void lowl_bloomfilter_write(lowl_bloomfilter* f, FILE* fp)
    #void lowl_bloomfilter_read(lowl_bloomfilter* f, FILE* fp)
