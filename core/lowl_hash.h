#ifndef LOWLHASH_H
#define LOWLHASH_H

/* data structures and functions related to hash tables.
	This includes:
		hash functions
		hash tables
		resizable arrays for use in hash tables
	*/

/********************************************************
 *							*
 *	 Begin temporarily-deprecated code.		*
 *							*
 ********************************************************/

/* generic Hash Function. */
// unsigned int HashFunction(void*)

/* multiply-shift and multiply-add-shift are a pair of near-universal
	and universal (respectively) hash functions. See
	http://en.wikipedia.org/wiki/Universal_hashing,
	particularly the section on hashing integers.

    These functions require knowing:
	M:	the base-2 log of the number m of bins into which
		we are hashing, i.e., m=2^M, where M is an int,
	w:	the number of bits in a machine word (w). 
*/
//unsigned int LOWLHASHES_GLOB_w;
//unsigned int LOWLHASHES_GLOB_M;

/* multiply-shift, described in wiki on universal hashing */
//unsigned int multip_shift(int x);
/* uses a seed value a, a non-negative odd integer < 2^w,
	where w is the size of a machine word (presumably w=32). */
//unsigned int LOWLHASHES_GLOB_MULTIPSHIFT_a;

/* A dummy function to allow for more generic code in hash tables. */
//unsigned int multip_shift_helper(void* vptr);

/* multiply-add-shift, described in wiki on universal hashing */
//unsigned int multip_add_shift_2(int x);
/* uses two seeds, a and b.
	a is a non-negative odd integer < 2^w,
	b is a positive non-negative integer 0<b<2^(w-M)
	where
	w is the number of bits in a machine word (e.g., w=32)
	M is such that m = 2^M, where m is the number of "bins".  */
//unsigned int LOWLHASHES_GLOB_MULTIPADDSHIFT_a;
//unsigned int LOWLHASHES_GLOB_MULTIPADDSHIFT_b;

/* A dummy function to allow for more generic code in hash tables. */
//unsigned int multip_add_shift_helper(void* vptr);

/********************************************************
 *							*
 *	End temporarily-deprecated code.		*
 *							*
 ********************************************************/



/********************************************************
 *							*
 *	Hash functions.					*
 *							*
 ********************************************************/

/* let's do the struct-based one in parallel, so we can choose which one we
	prefer down the line. */
typedef struct lowl_key_hash{
    unsigned long a;
    unsigned long b;
    unsigned int w;
    unsigned int M;
} lowl_key_hash;

lowl_hashoutput multip_add_shift( lowl_key x, lowl_key_hash* lihash);


/********************************************************
 *							*
 *	Resizable array for use in hash tables.		*
 *							*
 ********************************************************/

/* resizable array of lowl_keys, for use in hash tables. */
typedef struct lowl_rarr{
  unsigned int capacity; /* total number of slots available */
  unsigned int size; /* number of slots currently in use. */
  lowl_count* array;
}lowl_rarr;

int lowl_rarr_init(lowl_rarr* lr, unsigned int cap);

int lowl_rarr_insert(lowl_rarr* lr, unsigned int loc, lowl_count elmt);

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
