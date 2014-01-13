#include <stdlib.h>
#include <string.h>
#include "lowl_hash.h"
#include "lowl_sketch.h"

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
  int i;
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
  int i;
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
  int i;
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
  int i;
  for( i=0; i < cm->depth; i++ ) {
    memset( cm->counters+i, 0, cm->width*sizeof(unsigned int) );
  }
}

void cmsketch_destroy( cmsketch* cm ) {
  free( cm->hashes );
  int i;
  for( i=0; i<cm->depth; i++ ) {
    free( cm->counters+i );
  }
  return;
}
