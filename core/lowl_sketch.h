#ifndef LOWLSKETCH_H
#define LOWLSKETCH_H

#include <stdint.h>

/* functions and structures related to counting and sketching algorithms. */

typedef struct cmsketch{
  unsigned int width;
  unsigned int depth;
  /* we maintain width*depth counters, which we will store as an array
	of arrays (rather than as one contiguous block of memory,
	at least for now.	*/
  unsigned int** counters;
  motrag_hash* hashes; /* need an array of depth different hashes. */
}cmsketch; 

int cmsketch_init( cmsketch* cm, unsigned int m,
			unsigned int w, unsigned int d);
int cmsketch_update( cmsketch* cm, unsigned int i, unsigned int c );
int cmsketch_count( cmsketch* cm, unsigned int token );
void cmsketch_clear( cmsketch* cm );
void cmsketch_destroy( cmsketch* cm );


typedef struct bloomfilter { 
  unsigned int size;
  unsigned int k; /* number of hash functions to use. */
  uint32_t* b;
  lowl_key_hash* hash_key_to_word;
  lowl_key_hash* hash_key_to_bit;
  uint32_t* mask;
}bloomfilter;

// Self-explanatory functions
int bloomfilter_init(bloomfilter* f, size_t size, unsigned int k);
void bloomfilter_setmask( uint32_t* mask );
void bloomfilter_insertKey(bloomfilter* f, lowl_key k);
void bloomfilter_insertString(bloomfilter* f, char* x, int len);
int  bloomfilter_queryKey(bloomfilter* f, lowl_key k);
int  bloomfilter_queryString(bloomfilter* f, char* x, int len);
void bloomfilter_print(bloomfilter* f);
void bloomfilter_write(bloomfilter* f, FILE* fp);
void bloomfilter_read(bloomfilter* f, FILE* fp);
void bloomfilter_destroy(bloomfilter* f);

#endif
