#ifndef LOWLHASH_H
#define LOWLHASH_H

#include "lowl_types.h"

#define CUCKOO_NHASHES 8
#define CUCKOO_REHASH -2

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

lowl_hashoutput mod_fnv(const char *data, size_t len, unsigned int salt);

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
typedef struct motrag_hash{
  unsigned int m; // cardinality of the input universe
  unsigned int n; // cardinality of the output universe
  unsigned int p; // prime, p >= m.
  unsigned int a; // a \in {0,1,...,p-1}
  unsigned int b; // b \in {0,1,...,p-1}
}motrag_hash;

int motrag_hash_init( motrag_hash* lmh,
				unsigned int m, unsigned int n );

unsigned int motrag_map( unsigned int input, motrag_hash* lmh);

void motrag_hash_arm( motrag_hash* lmh );



/********************************************************
 *							*
 *	Resizable array for use in hash tables.		*
 *							*
 ********************************************************/

/* resizable array, for use in hash tables. */
typedef struct rarr{
  unsigned int capacity; /* total number of slots available */
  rarr_entry* array;
}rarr;

int rarr_init(rarr* lr, unsigned int cap);

int rarr_set(rarr* lr, unsigned int loc, rarr_entry* elmt);

int rarr_get(rarr* lr, unsigned int loc, rarr_entry* elmt);

int rarr_upsize(rarr* lr);

int rarr_downsize(rarr* lr);

int rarr_destroy(rarr* lr);

/********************************************************
 *							*
 *	Hash table for <lowl_key,lowl_count>		*
 *							*
 ********************************************************/
typedef struct ht_key_to_count{
  lowl_key_hash* hashfn;
  rarr* table;
  unsigned int size; /* number of elements we've inserted. */
}ht_key_to_count;

int ht_key_to_count_init( ht_key_to_count* ht, unsigned int capacity );
int ht_key_to_count_set( ht_key_to_count* ht, lowl_key key, lowl_count val);
int ht_key_to_count_get( ht_key_to_count* ht, lowl_key key, lowl_count* val);
void ht_key_to_count_clear( ht_key_to_count* ht );
void ht_key_to_count_destroy( ht_key_to_count* ht );


/********************************************************
 *							*
 *	Hash table for <lowl_key, string>		*
 *							*
 ********************************************************/


/********************************************************
 *							*
 *	Hash table for <string, lowl_count>		*
 *							*
 ********************************************************/

#endif
