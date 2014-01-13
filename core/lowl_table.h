#include <stddef.h>

static const size_t T_CAPACITY = 32;

typedef struct {
    char **data;
    size_t size;
} Table;

size_t tableSize(Table* t);
size_t tableInsert(Table* t, char* s);
char* tableGet(Table* t, size_t i);
Table* tableNew();
void tableDel(Table* t);
