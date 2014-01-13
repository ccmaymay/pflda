/* Bloom filter implementation        */
/* Author: Ashwin Lall                  */
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "lowl_hash.h"
//#include "GeneralHashFunction.c"

typedef struct lowl_bloomfilter
{ 
  int size;
  int k;
  uint32_t* b;
  struct GeneralHashFunction h;
  uint32_t* mask;
}lowl_bloomfilter;

// Self-explanatory functions
void lowl_bloomfilter_init(lowl_bloomfilter* f, int size, int k);
void lowl_bloomfilter_insertKey(lowl_bloomfilter* f, lowl_key k);
void lowl_bloomfilter_insertString(lowl_bloomfilter* f, char* x, int len);
int lowl_bloomfilter_queryKey(lowl_bloomfilter* f, lowl_key k);
int lowl_bloomfilter_queryString(lowl_bloomfilter* f, char* x, int len);
void lowl_bloomfilter_print(lowl_bloomfilter* f);
void lowl_bloomfilter_write(lowl_bloomfilter* f, FILE* fp);
void lowl_bloomfilter_read(lowl_bloomfilter* f, FILE* fp);
void lowl_bloomfilter_destroy(lowl_bloomfilter* f);
