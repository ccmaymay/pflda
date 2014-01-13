/* Bloom Filter implementation          */
/* Author: Ashwin Lall                  */
#include <math.h>
#include <stdint.h>
#include "lowl_bloomfilter.h"

void lowl_bloomfilter_init( lowl_bloomfilter* f,
			size_t size, unsigned int k  ) { 
  int i;
  // These masks help us to access the i-th bit in
  // a uint32_t counter (for i=1...32)
  f->mask = (uint32_t*)malloc( 32*sizeof(uint32_t) );
  for(i = 0; i < 8; ++i) { 
    f->mask[4 * i + 0] = (1 << 4 * i + 0);
    f->mask[4 * i + 1] = (1 << 4 * i + 1);
    f->mask[4 * i + 2] = (1 << 4 * i + 2);
    f->mask[4 * i + 3] = (1 << 4 * i + 3);
  }

  // we need size/sizeof(f->b) since each contributes sizeof(f->b) bytes
  f->size = size/sizeof(*(f->b)) + 1;

  f->k = k;
  f->b = (uint32_t*)malloc(f->size * sizeof(uint32_t));

  memset(f->b, 0, f->size * sizeof(uint32_t));

  f->hash_key_to_word = (lowl_key_hash**)malloc( f->k*sizeof(lowl_key_hash*) );
  f->hash_key_to_bit = (lowl_key_hash**)malloc( f->k*sizeof(lowl_key_hash*) );
  for( i=0; i<f->k; i++ ) {
    (f->hash_key_to_word)[i] = malloc( sizeof(lowl_key_hash) );
    (f->hash_key_to_bit)[i] = malloc( sizeof(lowl_key_hash) );
    lowl_key_hash_init( f->hash_key_to_word + i,
                        (unsigned int) 8*sizeof(lowl_key),
                        (unsigned int) 8*f->size );
    /* this hash function maps a lowl_key to a specific bit within a word. */
    lowl_key_hash_init( f->hash_key_to_bit + i,
                        (unsigned int) 8*sizeof(lowl_key),
                        (unsigned int) log2( 8*sizeof(*(f->b)) ) );
    /* seed the hashes. */
    lowl_key_hash_arm( f->hash_key_to_word + i );
    lowl_key_hash_arm( f->hash_key_to_bit + i );
  }
}

void lowl_bloomfilter_insertKey(lowl_bloomfilter* f, lowl_key key) { 
  int i;
  lowl_hashoutput word,bit;

  for(i = 0; i < f->k; ++i) { 
    ////lowl_key k->h), y, strlen(y));
    word = multip_add_shift( key, f->hash_key_to_word + i );
    bit = multip_add_shift( key, f->hash_key_to_bit + i );

    f->b[word] = f->b[word] | f->mask[bit];
  }
}

int lowl_bloomfilter_queryKey(lowl_bloomfilter* f, lowl_key key ) {
  int i;
  lowl_hashoutput word,bit;

  for(i = 0; i < f->k; ++i) {
    word = multip_add_shift( f->hash_key_to_word + i, key );
    bit = multip_add_shift( f->hash_key_to_bit + i, key );
    if ( (f->b[word] & f->mask[bit]) == 0)
      return 0;
  }

  return 1;
}

/***************************************************************
 *
 *	The following code is commented out until we are ready to
 *	hash strings inside of core.
 *
 ****************************************************************/

//void lowl_bloomfilter_insertString(lowl_bloomfilter* f, char* x, int len) { 
//  int i, index;
//  char y[len + 8];  // assumes that f->k < 10^8
//
//  for(i = 0; i < f->k; ++i) { 
//    sprintf(y, "%d%s", i, x);
//    index = hash(&(f->h), y, strlen(y));
//
//    f->b[index/32] = f->b[index/32] | f->mask[index % 32];
//  }
//}

//int lowl_bloomfilter_queryString(lowl_bloomfilter* f, char* x, int len) { 
//  int i, index;
//  char y[len + 8];  // assumes that f->k < 10^8
//
//  for(i = 0; i < f->k; ++i) { 
//    sprintf(y, "%d%s", i, x);
//    index = hash(&(f->h), y, strlen(y));
//    if ((f->b[index/32] & f->mask[index % 32]) == 0)
//      return 0;
//  }
//
//  return 1;
//}
/* End. */


void lowl_bloomfilter_print(lowl_bloomfilter* f) {
  int i, j;
  for(i = 0; i < f->size; ++i)
    for(j = 0; j < 32; ++j)
      if ((f->b[i] & f->mask[j]) == 0)
        printf("0");
      else
        printf("1");
  printf("\n");
}

void lowl_bloomfilter_write(lowl_bloomfilter* f, FILE* fp) {
  fwrite(&(f->size), sizeof(int), 1, fp);
  fwrite(&(f->k), sizeof(int), 1, fp);
  fwrite(f->b, sizeof(uint32_t), f->size, fp);
  writeHashFunction(&(f->h), fp);
}

void lowl_bloomfilter_read(lowl_bloomfilter* f, FILE* fp) {

  int i;

  f->mask = (uint32_t*) malloc(32 * sizeof(uint32_t));
  for(i = 0; i < 8; ++i)
    {
      f->mask[4 * i + 0] = (1 << 4 * i + 0);
      f->mask[4 * i + 1] = (1 << 4 * i + 1);
      f->mask[4 * i + 2] = (1 << 4 * i + 2);
      f->mask[4 * i + 3] = (1 << 4 * i + 3);
    }

  fread(&(f->size), sizeof(int), 1, fp);
  fread(&(f->k), sizeof(int), 1, fp);
  f->b = (uint32_t*)malloc(f->size * sizeof(uint32_t));
  fread(f->b, sizeof(uint32_t), f->size, fp);
  readHashFunction(&(f->h), fp);
}

void lowl_bloomfilter_destroy(lowl_bloomfilter* f) {
  free(f->b);
  free(f->mask);
  free(f->hash_key_to_word);
  free(f->hash_key_to_bit);
}
