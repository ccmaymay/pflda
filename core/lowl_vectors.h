#ifndef LOWLVECTORS_H
#define LOWLVECTORS_H

#include "lowl_types.h"

typedef struct bitvector{
  unsigned int numbits;
  char* bits;
}bitvector;

int bitvector_init( bitvector* bv, unsigned int numbits);
void bitvector_find_indices( 	unsigned int loc,
				unsigned int* charindex,
				unsigned int* bitindex );
int bitvector_on( bitvector* bv, unsigned int loc);
int bitvector_off( bitvector* bv, unsigned int loc);
int bitvector_flip( bitvector* bv, unsigned int loc);
int bitvector_lookup( bitvector* bv, unsigned int loc);
void bitvector_clear( bitvector* bv );   
void bitvector_destroy( bitvector* bv );

typedef struct vector{
  unsigned int length;
  float* components;
}vector;

typedef struct svec_entry{
  unsigned int component;
  float value;
}svec_entry;

typedef struct sparse_vector{
  unsigned int length;
  svec_entry* entries;
}sparse_vector;

#endif
