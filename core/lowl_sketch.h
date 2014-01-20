#ifndef LOWLSKETCH_H
#define LOWLSKETCH_H

#include <stdint.h>
#include "lowl_types.h"
#include "lowl_hash.h"

/* functions and structures related to counting and sketching algorithms. */

/* count-min sketch. */
typedef struct cmsketch{
  size_t width;
  size_t depth;
  /* we maintain width*depth counters, which we will store as an array
	of arrays (rather than as one contiguous block of memory,
	at least for now.	*/
  lowl_count** counters;
  char_hash* hashes; /* need an array of depth different hashes. */
}cmsketch; 

int cmsketch_init(cmsketch* cm, size_t w, size_t d);
int cmsketch_add(cmsketch* cm, const char *x, size_t len, lowl_count delta);
lowl_count cmsketch_query(cmsketch* cm, const char *x, size_t len);
void cmsketch_print(cmsketch* cm);
void cmsketch_write(cmsketch* cm, FILE* fp);
int cmsketch_read(cmsketch* cm, FILE* fp);
void cmsketch_clear(cmsketch* cm);
void cmsketch_destroy(cmsketch* cm);

/* bloom filter. */
typedef struct bloomfilter {
  unsigned int size;
  unsigned int k; /* number of hash functions to use. */
  uint32_t* b;
  /* we're using a clever trick whereby we get k nearly-independent
	hash functions using only two. */
  char_hash hash_key_to_word1;
  char_hash hash_key_to_word2;
  char_hash hash_key_to_bit1;
  char_hash hash_key_to_bit2;
  uint32_t* mask;
}bloomfilter;

int bloomfilter_init(bloomfilter* f, size_t numbytes, unsigned int k);
void bloomfilter_setmask( uint32_t* mask );
void bloomfilter_insert(bloomfilter* f, const char* x, size_t len);
int bloomfilter_query(bloomfilter* f, const char* x, size_t len);
void bloomfilter_print(bloomfilter* f);
void bloomfilter_write(bloomfilter* f, FILE* fp);
int bloomfilter_read(bloomfilter* f, FILE* fp);
void bloomfilter_destroy(bloomfilter* f);
//lowl_hashoutput bloomfilter_hash2word(bloomfilter* f,
//					unsigned int i, lowl_key key );
//lowl_hashoutput bloomfilter_hash2bit(bloomfilter* f,
//                                        unsigned int i, lowl_key key );

/* bloomier filter. */
//typedef struct bloomierfilter{
//  unsigned int size; /* number of (fingerprint,value) pairs to store. */
//  motrag_hash* hash_to_fingerprint;
//  motrag_hash* hash_to_cell;
//  linkedlist* overflow;
//}bloomierfilter;

//int bloomierfilter_init( bloomierfilter* f, unsigned int size );
//void bloomierfilter_set(bloomierfilter* f, lowl_key k, void* data);
//void bloomierfilter_get(bloomierfilter* f, lowl_key k, void* result);
//void bloomierfilter_destroy(bloomierfilter* f);

#endif
