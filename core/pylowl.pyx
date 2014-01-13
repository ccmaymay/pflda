cimport lowl

cdef class Table:
    cdef lowl.Table* _table

    def __str__(self):
        lines = []
        for i in range(len(self)):
            lines.append('%d: %s' % (i, self[i]))
        return '\n'.join(lines)

    def __len__(self):
        return lowl.tableSize(self._table)

    def __getitem__(self, i):
        cdef char *s
        cStr = lowl.tableGet(self._table, i)
        if cStr is NULL:
            raise Exception("Illegal index %d" % i)
        pyStr = cStr
        return pyStr

    def insert(self, s):
        i = lowl.tableInsert(self._table, s)
        if i > 1e6:
            raise Exception("Insert may or may not have failed")
        return i

    def __cinit__(self):
        self._table = lowl.tableNew()
        if self._table is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._table is not NULL:
            lowl.tableDel(self._table)
