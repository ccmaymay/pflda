#ifndef LOWLSKETCH_H
#define LOWLSKETCH_H

/* functions and structures related to counting and sketching algorithms. */

typedef struct lowl_cmsketch{
  unsigned int width;
  unsigned int depth;
  /* we maintain width*depth counters, which we will store as an array
	of arrays (rather than as one contiguous block of memory,
	at least for now.	*/
  unsigned int** counters;
  lowl_motrag_hash* hashes; /* need an array of depth different hashes. */
}lowl_cmsketch; 

int lowl_cmsketch_init( lowl_cmsketch* cm, unsigned int m,
			unsigned int w, unsigned int d);

int lowl_cmsketch_update( lowl_cmsketch* cm, unsigned int i, unsigned int c );

int lowl_cmsketch_count( lowl_cmsketch* cm, unsigned int token );

void lowl_cmsketch_clear( lowl_cmsketch* cm );

void lowl_cmsketch_destroy( lowl_cmsketch* cm );

#endif
