#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "lowl_hash.h"
#include "lowl_math.h"
#include "lowl_sketch.h"

/*********************************************************
 *                                                       *
 *        Count-min sketch.                              *
 *                                                       *
 *********************************************************/


/* initial setup of cmsketch. */
int cmsketch_init(cmsketch* cm, size_t w, size_t d) {
  /* w is the desired width of the sketch
	 * d is the desired depth of the sketch.
	 * See http://dimacs.rutgers.edu/~graham/pubs/papers/cmencyc.pdf
	 * for a good overview of the CM sketch.
   */
  cm->width = w;
  cm->depth = d;

  cm->counters = malloc(d * sizeof(lowl_count*));
  if (cm->counters == NULL)
    return LOWLERR_BADMALLOC;
  for (size_t i = 0; i < d; ++i) {
    cm->counters[i] = malloc(w * sizeof(lowl_count));
    if (cm->counters[i] == NULL)
      return LOWLERR_BADMALLOC;
    for (size_t j = 0; j < w; ++j)
      cm->counters[i][j] = 0;
  }

  char_hash_arm(&(cm->hash_key1));
  char_hash_arm(&(cm->hash_key2));

  return LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS;
}

void cmsketch_add(cmsketch* cm, const char *x, size_t len, lowl_count delta) {
  lowl_hashoutput hash1, hash2;
  hash1 = mod_fnv(x, len, &(cm->hash_key1));
  hash2 = mod_fnv(x, len, &(cm->hash_key2));
  for (size_t i = 0; i < cm->depth; ++i)
    cm->counters[i][(hash1 + i*hash2) % cm->width] += delta;
}

lowl_count cmsketch_query(cmsketch* cm, const char *x, size_t len) {
  lowl_hashoutput hash1, hash2;
  hash1 = mod_fnv(x, len, &(cm->hash_key1));
  hash2 = mod_fnv(x, len, &(cm->hash_key2));
  lowl_count c = cm->counters[0][hash1 % cm->width];
  for (size_t i = 1; i < cm->depth; ++i)
    c = min(c, cm->counters[i][(hash1 + i*hash2) % cm->width]);
  return c;
}

void cmsketch_print(cmsketch* cm) {
  for (size_t i = 0; i < cm->depth; ++i) {
    printf("%u", (unsigned int) cm->counters[i][0]);
    for (size_t j = 1; j < cm->width; ++j)
      printf(" %u", (unsigned int) cm->counters[i][j]);
    printf("\n");
  }
}

void cmsketch_write(cmsketch* cm, FILE* fp) {
  fwrite(&(cm->width), sizeof(size_t), 1, fp);
  fwrite(&(cm->depth), sizeof(size_t), 1, fp);
  fwrite(&(cm->hash_key1), sizeof(char_hash), 1, fp);
  fwrite(&(cm->hash_key2), sizeof(char_hash), 1, fp);
}

int cmsketch_read(cmsketch* cm, FILE* fp) {
  fread(&(cm->width), sizeof(size_t), 1, fp);
  fread(&(cm->depth), sizeof(size_t), 1, fp);
  fread(&(cm->hash_key1), sizeof(char_hash), 1, fp);
  fread(&(cm->hash_key2), sizeof(char_hash), 1, fp);

  cm->counters = malloc(cm->depth * sizeof(lowl_count*));
  if (cm->counters == NULL)
    return LOWLERR_BADMALLOC;
  for (size_t i = 0; i < cm->depth; ++i) {
    cm->counters[i] = malloc(cm->width * sizeof(lowl_count));
    if (cm->counters[i] == NULL)
      return LOWLERR_BADMALLOC;
    for (size_t j = 0; j < cm->width; ++j)
      cm->counters[i][j] = 0;
  }

  return LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS;
}

void cmsketch_clear(cmsketch* cm) {
  /* zero all counters. */
  for (size_t i = 0; i < cm->depth; ++i) {
    for (size_t j = 0; j < cm->width; ++j) {
      cm->counters[i][j] = 0;
    }
  }
}

void cmsketch_destroy(cmsketch* cm) {
  for (size_t i = 0; i < cm->depth; ++i) {
    if (cm->counters[i] != NULL)
      free(cm->counters[i]);
    cm->counters[i] = NULL;
  }

  if (cm->counters != NULL)
    free(cm->counters);
  cm->counters = NULL;
}

/*********************************************************
 *                                                       *
 *        Bloom filter                                   *
 *    Based on code by Ben Van Durme and Ashwin Lall     *
 *                                                       *
 *********************************************************/

int bloomfilter_init(bloomfilter* f, size_t numbytes, unsigned int k) {
  /* numbytes is the number of bytes to use in the bloom filter.
	k is the number of hash functions.

	Allocate necessary memory, initalize and arm hash functions,
	return a success code. */

  // Need to add code that ensures that size is a power of 2.

  // These masks help us to access the i-th bit in
  // a uint32_t counter (for i=1...32)
  f->mask = (uint32_t*)malloc( 32*sizeof(uint32_t) );
  if( f->mask==NULL ) {
    return LOWLERR_BADMALLOC;
  }
  bloomfilter_setmask( f->mask );

  // we need size/sizeof(f->b) since each contributes sizeof(f->b) bytes
  f->size = numbytes/sizeof(*(f->b));
  if( f->size == 0 ) {
    f->size = 1; /* if size was too small to be useful. */
  }

  f->k = k;
  f->b = (uint32_t*)malloc(f->size * sizeof(uint32_t));

  memset(f->b, 0, f->size * sizeof(uint32_t));

  if( f->b==NULL ) {
    return LOWLERR_BADMALLOC;
  }

  char_hash_arm(&(f->hash_key_to_word1));
  char_hash_arm(&(f->hash_key_to_word2));
  char_hash_arm(&(f->hash_key_to_bit1));
  char_hash_arm(&(f->hash_key_to_bit2));

  return LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS;
}

void bloomfilter_insert(bloomfilter* f, const char* x, size_t len) {
  const size_t bits_per_bf_word = 8*sizeof(*(f->b));
  lowl_hashoutput word,bit,hash2word1,hash2word2,hash2bit1,hash2bit2;

  /* we use a scheme whereby two hashes give rise to k approximately
        independent hashes, where hash function h_i is given by
        h_i(x) = f(x) + i*g(x). */
  hash2word1 = mod_fnv(x, len, &(f->hash_key_to_word1));
  hash2word2 = mod_fnv(x, len, &(f->hash_key_to_word2));
  hash2bit1 = mod_fnv(x, len, &(f->hash_key_to_bit1));
  hash2bit2 = mod_fnv(x, len, &(f->hash_key_to_bit2));

  for (unsigned int i = 0; i < f->k; ++i) {
    word = (hash2word1 + i*hash2word2) % f->size;
    bit = (hash2bit1 + i*hash2bit2) % bits_per_bf_word;
    
    f->b[word] |= f->mask[bit];
  }
}

lowl_bool bloomfilter_query(bloomfilter* f, const char* x, size_t len) {
  const size_t bits_per_bf_word = 8*sizeof(*(f->b));
  lowl_hashoutput word,bit,hash2word1,hash2word2,hash2bit1,hash2bit2;

  /* we use a scheme whereby two hashes give rise to k approximately
        independent hashes, where hash function h_i is given by
        h_i(x) = f(x) + i*g(x). */
  hash2word1 = mod_fnv(x, len, &(f->hash_key_to_word1));
  hash2word2 = mod_fnv(x, len, &(f->hash_key_to_word2));
  hash2bit1 = mod_fnv(x, len, &(f->hash_key_to_bit1));
  hash2bit2 = mod_fnv(x, len, &(f->hash_key_to_bit2));

  for (unsigned int i = 0; i < f->k; ++i) {
    word = (hash2word1 + i*hash2word2) % f->size;
    bit = (hash2bit1 + i*hash2bit2) % bits_per_bf_word;
    
    if ((f->b[word] & f->mask[bit]) == 0)
      return FALSE;
  }

  return TRUE;
}

void bloomfilter_print(bloomfilter* f) {
  unsigned int i,j;
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
  fwrite( &(f->hash_key_to_word1), sizeof( char_hash ), 1, fp);
  fwrite( &(f->hash_key_to_word2), sizeof( char_hash ), 1, fp);
  fwrite( &(f->hash_key_to_bit1), sizeof( char_hash ), 1, fp); 
  fwrite( &(f->hash_key_to_bit2), sizeof( char_hash ), 1, fp); 
}

int bloomfilter_read(bloomfilter* f, FILE* fp) {
  f->mask = (uint32_t*) malloc(32 * sizeof(uint32_t));
  bloomfilter_setmask( f->mask );

  fread(&(f->size), sizeof(unsigned int), 1, fp);
  fread(&(f->k), sizeof(unsigned int), 1, fp);
  f->b = (uint32_t*)malloc( f->size*sizeof(uint32_t));
  if (f->b == 0) return LOWLERR_BADMALLOC;
  fread(f->b, sizeof(uint32_t), f->size, fp);
  fread( &(f->hash_key_to_word1), sizeof( char_hash ), 1, fp);
  fread( &(f->hash_key_to_word2), sizeof( char_hash ), 1, fp);
  fread( &(f->hash_key_to_bit1), sizeof( char_hash ), 1, fp);
  fread( &(f->hash_key_to_bit2), sizeof( char_hash ), 1, fp);
  return LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS;
}

void bloomfilter_destroy(bloomfilter* f) {
  if (f->b != NULL)
    free(f->b);
  f->b = NULL;

  if (f->mask != NULL)
    free(f->mask);
  f->mask = NULL;
}

void bloomfilter_setmask( uint32_t* mask ) {
  int i;
  for(i = 0; i < 8; ++i) {
    mask[4 * i + 0] = (1 << (4 * i + 0));
    mask[4 * i + 1] = (1 << (4 * i + 1));
    mask[4 * i + 2] = (1 << (4 * i + 2));
    mask[4 * i + 3] = (1 << (4 * i + 3));
  }
}

/*********************************************************
 *                                                       *
 *        Bloomier Filter.                               *
 *                                                       *
 *********************************************************/

// coming soon.
