#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include "lowl_hash.h"
#include "lowl_sketch.h"

/*********************************************************
 *                                                       *
 *        Count-min sketch.                              *
 *                                                       *
 *********************************************************/


/* initial setup of cmsketch. */
int cmsketch_init( cmsketch* cm, unsigned int m,
			unsigned int w, unsigned int d) {
  /* unsigned int m is the cardinality of the input universe.
	unsigned int w is the desired width of the sketch
	unsigned int d is the desired depth of the sketch.
	See http://dimacs.rutgers.edu/~graham/pubs/papers/cmencyc.pdf
	for a good overview of the CM sketch.	*/
  cm->width = w;
  cm->depth = d;
  cm->counters = malloc( d*sizeof(unsigned int*) );
  unsigned int i;
  for( i=0; i < d; i++ ) {
    (cm->counters)[i] = malloc( w*sizeof(unsigned int) );
  }
  cm->hashes = malloc( d*sizeof( motrag_hash ) );
  int succ;
  for( i=0; i<d; i++ ) {
    succ = motrag_hash_init( cm->hashes+i, m, w );
    if( succ != 0 ) {
      return -1;
    }
  }
  return 0;
}

void cmsketch_arm( cmsketch* cm ) {
  /* seed all the hashes.	*/
  unsigned int i;
  for( i=0; i< cm->depth; i++ ) {
    motrag_hash_arm( cm->hashes+i );
  }
  return;
}

int cmsketch_update(cmsketch* cm, unsigned int elmt,
				unsigned int c) {
  if( elmt > ( cm->hashes )->m ) { /* element is out of range. */
    return -1;
  }
  unsigned int i;
  unsigned int hashoutput;
  for( i=0; i<cm->depth; i++ ) {
    /* the i-th hash function, evaluated on the input element, determines
	which counter in the i-th array gets added to.	*/
    hashoutput = motrag_map( elmt, cm->hashes+i );
    (cm->counters+i)[hashoutput] += c;
  }
  return 0;
}

int cmsketch_count( cmsketch* cm, unsigned int elmt ) {
  /* count the given element as having occurred once.	*/
  return cmsketch_update( cm, elmt, 1);
}

void cmsketch_clear( cmsketch* cm ) {
  /* zero all counters. */
  unsigned int i;
  for( i=0; i < cm->depth; i++ ) {
    memset( cm->counters+i, 0, cm->width*sizeof(unsigned int) );
  }
}

void cmsketch_destroy( cmsketch* cm ) {
  free( cm->hashes );
  cm->hashes = NULL;
  unsigned int i;
  for( i=0; i<cm->depth; i++ ) {
    free( cm->counters+i );
    (cm->counters)[i] = NULL;
  }
  return;
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
    return -1; /* allocation failed. */
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
    return -1; /* allocation failed. */
  }

  char_hash_arm(&(f->hash_key_to_word1));
  char_hash_arm(&(f->hash_key_to_word2));
  char_hash_arm(&(f->hash_key_to_bit1));
  char_hash_arm(&(f->hash_key_to_bit2));

  return 0;
}

void lowl_bloomfilter_insert(bloomfilter* f, const char* x, size_t len) { 
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

bool lowl_bloomfilter_query(bloomfilter* f, const char* x, size_t len) { 
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
      return 0;
  }

  return 1;
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

void bloomfilter_read(bloomfilter* f, FILE* fp) {
  f->mask = (uint32_t*) malloc(32 * sizeof(uint32_t));
  bloomfilter_setmask( f->mask );

  fread(&(f->size), sizeof(int), 1, fp);
  fread(&(f->k), sizeof(int), 1, fp);
  f->b = (uint32_t*)malloc( f->size*sizeof(uint32_t));
  fread(f->b, sizeof(uint32_t), f->size, fp);
  fread( &(f->hash_key_to_word1), sizeof( char_hash ), 1, fp);
  fread( &(f->hash_key_to_word2), sizeof( char_hash ), 1, fp);
  fread( &(f->hash_key_to_bit1), sizeof( char_hash ), 1, fp);
  fread( &(f->hash_key_to_bit2), sizeof( char_hash ), 1, fp);
}

void bloomfilter_destroy(bloomfilter* f) {
  free(f->b);
  free(f->mask);

  f->b = NULL;
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
  return;
}

/*********************************************************
 *                                                       *
 *        Bloomier Filter.                               *
 *                                                       *
 *********************************************************/

// coming soon.
