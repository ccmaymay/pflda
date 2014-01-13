/* Bloom filter implementation        */
/* Author: Ashwin Lall                  */
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "lowl_hash.h"
//#include "GeneralHashFunction.c"

typedef struct bloomfilter { 
  unsigned int size;
  unsigned int k; /* number of hash functions to use. */
  uint32_t* b;
  lowl_key_hash* hash_key_to_word;
  lowl_key_hash* hash_key_to_bit;
  uint32_t* mask;
}bloomfilter;

// Self-explanatory functions
void bloomfilter_init(bloomfilter* f, size_t size, unsigned int k);
void bloomfilter_insertKey(bloomfilter* f, lowl_key k);
void bloomfilter_insertString(bloomfilter* f, char* x, int len);
int  bloomfilter_queryKey(bloomfilter* f, lowl_key k);
int  bloomfilter_queryString(bloomfilter* f, char* x, int len);
void bloomfilter_print(bloomfilter* f);
void bloomfilter_write(bloomfilter* f, FILE* fp);
void bloomfilter_read(bloomfilter* f, FILE* fp);
void bloomfilter_destroy(bloomfilter* f);
