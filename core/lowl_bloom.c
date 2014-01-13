/* Bloom Filter implementation          */
/* Author: Ashwin Lall                  */
#include <math.h>
#include <stdint.h>
#include "lowl_bloom.h"

void bloomfilter_init( bloomfilter* f,
			size_t numbytes, unsigned int k  ) { 
  /* numbytes is the number of bytes to use in the bloom filter.
	k is the number of hash functions. */

  // Need to add code that ensures that size is a power of 2.

  int i;
  // These masks help us to access the i-th bit in
  // a uint32_t counter (for i=1...32)
  f->mask = (uint32_t*)malloc( 32*sizeof(uint32_t) );
  for(i = 0; i < 8; ++i) { 
    f->mask[4 * i + 0] = (1 << (4 * i + 0));
    f->mask[4 * i + 1] = (1 << (4 * i + 1));
    f->mask[4 * i + 2] = (1 << (4 * i + 2));
    f->mask[4 * i + 3] = (1 << (4 * i + 3));
  }

  // we need size/sizeof(f->b) since each contributes sizeof(f->b) bytes
  f->size = numbytes/sizeof(*(f->b));
  if( f->size == 0 ) f->size = 1; /* if size was too small to be useful. */

  f->k = k;
  f->b = (uint32_t*)malloc(f->size * sizeof(uint32_t));

  memset(f->b, 0, f->size * sizeof(uint32_t));

  f->hash_key_to_word = (lowl_key_hash*)malloc( f->k*sizeof(lowl_key_hash) );
  f->hash_key_to_bit = (lowl_key_hash*)malloc( f->k*sizeof(lowl_key_hash) );
  for( i=0; i<f->k; i++ ) {
    lowl_key_hash_init( f->hash_key_to_word + i,
                        (unsigned int) 8*sizeof(lowl_key),
                        (unsigned int) log2( f->size ) );
    /* this hash function maps a lowl_key to a specific bit within a word. */
    lowl_key_hash_init( f->hash_key_to_bit + i,
                        (unsigned int) 8*sizeof(lowl_key),
                        (unsigned int) log2( 8*sizeof(*(f->b)) ) );
    /* seed the hashes. */
    lowl_key_hash_arm( f->hash_key_to_word + i );
    lowl_key_hash_arm( f->hash_key_to_bit + i );
  }
}

void bloomfilter_insertKey(bloomfilter* f, lowl_key key) { 
  int i;
  lowl_hashoutput word,bit;

  for(i = 0; i < f->k; ++i) { 
    ////lowl_key k->h), y, strlen(y));
    word = multip_add_shift( key, f->hash_key_to_word + i );
    bit = multip_add_shift( key, f->hash_key_to_bit + i );

    f->b[word] = f->b[word] | f->mask[bit];
  }
}

int bloomfilter_queryKey(bloomfilter* f, lowl_key key ) {
  int i;
  lowl_hashoutput word,bit;

  for(i = 0; i < f->k; ++i) {
    word = multip_add_shift( key, f->hash_key_to_word + i );
    bit = multip_add_shift( key, f->hash_key_to_bit + i );
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

void bloomfilter_print(bloomfilter* f) {
  int i, j;
  for(i = 0; i < f->size; ++i)
    for(j = 0; j < 8*sizeof(*(f->b)); ++j)
      if ((f->b[i] & f->mask[j]) == 0)
        printf("0");
      else
        printf("1");
  printf("\n");
}

void bloomfilter_write(bloomfilter* f, FILE* fp) {
  /* serialize the filter to the given file. */
  fwrite( &(f->size), sizeof(unsigned int), 1, fp);
  fwrite( &(f->k), sizeof(unsigned int), 1, fp);
  fwrite( f->b, sizeof( *(f->b) ), f->size, fp);
  fwrite( f->hash_key_to_word, sizeof( lowl_key_hash ), f->k, fp);
  fwrite( f->hash_key_to_bit, sizeof( lowl_key_hash ), f->k, fp); 
}

void bloomfilter_read(bloomfilter* f, FILE* fp) {

  int i;

  f->mask = (uint32_t*) malloc(32 * sizeof(uint32_t));
  for(i = 0; i < 8; ++i)
    {
      f->mask[4 * i + 0] = (1 << (4 * i + 0));
      f->mask[4 * i + 1] = (1 << (4 * i + 1));
      f->mask[4 * i + 2] = (1 << (4 * i + 2));
      f->mask[4 * i + 3] = (1 << (4 * i + 3));
    }

  fread(&(f->size), sizeof(int), 1, fp);
  fread(&(f->k), sizeof(int), 1, fp);
  f->b = (uint32_t*)malloc( f->size*sizeof(uint32_t));
  fread(f->b, sizeof(uint32_t), f->size, fp);
  fread(f->hash_key_to_word, sizeof( lowl_key_hash), f->k, fp);
  fread(f->hash_key_to_bit, sizeof( lowl_key_hash), f->k, fp);
}

void bloomfilter_destroy(bloomfilter* f) {
  free(f->b);
  free(f->mask);
  free(f->hash_key_to_word);
  free(f->hash_key_to_bit);
}
