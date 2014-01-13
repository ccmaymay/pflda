cdef extern from "lowl_table.h":
    ctypedef struct Table:
        pass

    size_t tableSize(Table* t)
    size_t tableInsert(Table* t, char* s)
    char* tableGet(Table* t, size_t i)
    Table* tableNew()
    void tableDel(Table* t)
