#ifndef LOWLHASH_H
#define LOWLHASH_H

#include "lowl_types.h"

/* data structures and functions related to hash tables.
	This includes:
		hash functions
		hash tables
		resizable arrays for use in hash tables
	*/


/********************************************************
 *							*
 *	Hash functions.					*
 *							*
 ********************************************************/

/* multiply-add-shift, described in wiki on universal hashing */
/* uses two seeds, a and b.
	a is a non-negative odd integer 0 < a < 2^w,
	b is a positive non-negative integer 0<b<2^(w-M)
	where
	w is the number of bits in a machine word (e.g., w=32)
	M is such that m = 2^M, where m is the number of "bins".  */
typedef struct lowl_key_hash{
    unsigned long a;
    unsigned long b;
    unsigned int w;
    unsigned int M;
} lowl_key_hash;

lowl_hashoutput multip_add_shift( lowl_key x, lowl_key_hash* lihash);

int lowl_key_hash_init( lowl_key_hash* lkh, unsigned int w, unsigned int M);

void lowl_key_hash_arm( lowl_key_hash* lkh );

/* 2-universal hash function, from Motwani and Raghavan's randomized
	algorithms textbook, for use with sketching algorithms, primarily.
	Here we are hashing from a universe M to universe N with
	M = {0,1,...,m-1}
	N = {0,1,...,n-1}
	We choose a prime p >= m, and choose a,b \in \mathbb{Z}_p
		i.e., a,b \in {0,1,...,p-1}
	Define two functions:
	g_{a,b}(x) = ax + b (\mod p)
	f_{a,b}(x) = g_{a,b}( f_{a,b}(x) ).
	Our hash function is then given by
	h_{a,b}(x) = g_{a,b}(f_{a,b}(x)).
	This is a 2-universal hash function, meaning that hash functions
	are pair-wise independent.	*/
typedef struct lowl_motrag_hash{
  unsigned int m; // cardinality of the input universe
  unsigned int n; // cardinality of the output universe
  unsigned int p; // prime, p >= m.
  unsigned int a; // a \in {0,1,...,p-1}
  unsigned int b; // b \in {0,1,...,p-1}
}lowl_motrag_hash;

int lowl_motrag_hash_init( lowl_motrag_hash* lmh,
				unsigned int m, unsigned int n );

unsigned int lowl_motrag_map( unsigned int input, lowl_motrag_hash* lmh);

void lowl_motrag_hash_arm( lowl_motrag_hash* lmh );



/********************************************************
 *							*
 *	Resizable array for use in hash tables.		*
 *							*
 ********************************************************/

/* resizable array of lowl_keys, for use in hash tables. */
typedef struct lowl_rarr{
  unsigned int capacity; /* total number of slots available */
  lowl_count* array;
}lowl_rarr;

int lowl_rarr_init(lowl_rarr* lr, unsigned int cap);

int lowl_rarr_set(lowl_rarr* lr, unsigned int loc, lowl_count elmt);

int lowl_rarr_get(lowl_rarr* lr, unsigned int loc, lowl_count* elmt);

int lowl_rarr_upsize(lowl_rarr* lr);

int lowl_rarr_downsize(lowl_rarr* lr);

int lowl_rarr_destroy(lowl_rarr* lr);

/********************************************************
 *							*
 *	Hash table for storing counts			*
 *							*
 ********************************************************/



/********************************************************
 *							*
 *	Hash table for storing keys			*
 *							*
 ********************************************************/

// Not actually sure whether or not this will be necessary, yet.


#endif
