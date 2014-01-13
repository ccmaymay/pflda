#ifndef LOWLSKETCH_H
#define LOWLSKETCH_H

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

#endif
