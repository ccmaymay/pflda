#include "lowl_table.h"

#include <stdlib.h>

size_t tableSize(Table* t) {
    return t->size;
}

size_t tableInsert(Table* t, char* s) {
    if (t->size == T_CAPACITY) return (size_t) -1;
    size_t i = t->size;
    t->data[i] = s;
    ++(t->size);
    return i;
}

char* tableGet(Table* t, size_t i) {
    if (i >= t->size) return NULL;
    else return t->data[i];
}

Table* tableNew() {
    Table* t = malloc(sizeof(Table));
    if (t == 0) return NULL;
    t->data = malloc(T_CAPACITY * sizeof(char*));
    if (t->data == 0) return NULL;
    t->size = 0;
    return t;
}

void tableDel(Table* t) {
    free(t);
}
