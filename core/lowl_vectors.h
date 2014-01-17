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
int bitvector_set( bitvector* bv, char byte,
			unsigned int start, unsigned int nbits );
int bitvector_lookup( bitvector* bv, unsigned int loc);
void bitvector_clear( bitvector* bv );
void bitvector_print( bitvector* bv ); 
void bitvector_destroy( bitvector* bv );

typedef struct dense_vector{
  unsigned int length;
  float* entries;
}dense_vector;

typedef struct svec_entry{
  unsigned int component;
  float value;
}svec_entry;

typedef struct sparse_vector{
  unsigned int length; /* the actual dimensionality of the vector */
  /* the number of non-zero entries in the vector, i.e., the length of
	the array svec_entry* entries. */
  unsigned int sparsity;
  svec_entry* entries;
}sparse_vector;

/* a union that lets us refer generically to either a dense or
	sparse vector (but not to a bit vector) */
typedef union numeric_vector{
  sparse_vector sparse;
  dense_vector dense;
}numeric_vector;

int numeric_vector_is_sparse(numeric_vector* numvec); 
int numeric_vector_is_dense(numeric_vector* numvec); 
float numeric_vector_get_component(numeric_vector* numvec, unsigned int comp);
float dense_vector_get_component(dense_vector* denvec, unsigned int comp);
float sparse_vector_get_component(sparse_vector* spavec, unsigned int comp); 
float numeric_vector_dot_product(numeric_vector* nvaa, numeric_vector* nvbb);
float sparse_vector_dot_product(sparse_vector* svaa, sparse_vector* svbb);
float dense_vector_dot_product(dense_vector* dvaa, dense_vector* dvbb);
float sparsedense_vector_dot_product(sparse_vector* spa, dense_vector* den);




#endif
