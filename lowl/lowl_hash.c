#include "lowl_types.h"
#include "lowl_hash.h"


/* might want to change this to, say, 8*sizeof(long long), or something,
	depending on just how large we expect our hash tables to be. */
//LOWLHASHES_GLOB_w = 8*sizeof(int);

/* this is kind of silly, since the number of bins into which we are hashing
	will be different between different instances... */
//LOWLHASHES_GLOB_M = 8*sizeof(int);

/*
unsigned int multip_shift(int x) {
  return (unsigned int) (LOWLHASHES_MULTIPSHIFT_a*x)
	>> (LOWLHASHES_GLOB_w - LOWLHAHES_GLOB_M);
}

unsigned int multip_add_shift(int x) {
  return (unsigned int) (LOWLHASHES_MULTIPADDSHIFT_a*x
			+ LOWLHASHES_MULTIPADDSHIFT_b)
	>> (LOWLHASHES_GLOB_w - LOWLHAHES_GLOB_M);
} */

lowl_hashOutput multip_add_shift(lowl_key x, Lowl_Int_Hash* h) {
  return (lowl_hashOutput) ((h->a)*x + (h->b)) >> (h->w - h->M);
}
