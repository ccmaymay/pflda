#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lowl_types.h"
#include "lowl_math.h"
#include "lowl_hash.h"

/********************************************************
 *                                                      *
 *      Hash functions.					*
 *                                                      *
 ********************************************************/

/*
 * Modified FNV hash function that passes distribution tests and
 * achieves avalanche, from Jerboa:
 * $ git clone https://github.com/vandurme/jerboa.git
 * (see src/main/java/edu/jhu/jerboa/util/Hash.java)
 * 
 * Excerpt of original comment:
 *   I've (vandurme) played with many of the hash functions out there, and
 *   have found the following to give the most uniform results on data I've
 *   played with, both wrt randomness for the same key with different salts,
 *   and between keys that are very similar (such as "1" and "2") using the
 *   same salt. I've observed minor correlation between similar keys, very
 *   little or no correlation between the same key using different salts.
 *
 *   This is a modified version (to accept a salt) of Fig. 4 from
 *   {@linktourl http://home.comcast.net/~bretm/hash/6.html}
 */
lowl_hashoutput mod_fnv(const char *data, size_t len, char_hash* h) {
  unsigned int p = 16777619;
  lowl_hashoutput hash = 1315423911;
  hash = (hash ^ h->salt) * p;
  for (size_t i = 0; i < len; ++i)
    hash = (hash ^ data[i]) * p;
  hash += hash << 13;
  hash ^= hash >> 7;
  hash += hash << 3;
  hash ^= hash >> 17;
  hash += hash << 5;
  return hash;
}

void char_hash_arm(char_hash* ch) { 
  ch->salt = (unsigned int) random();
}

/* multiply-add-shift hash function. */
lowl_hashoutput multip_add_shift(lowl_key x, lowl_key_hash* h) {
  if( h->w < h->M ) {
    return (lowl_hashoutput) (h->a)*((unsigned int)x) + (h->b);
  } else {
    return ((lowl_hashoutput)
  	 ((h->a)*((unsigned int)x) + (h->b))) >> (h->w - h->M) ;
  }
}

int lowl_key_hash_init( lowl_key_hash* lkh, unsigned int w, unsigned int M) {
  if( w==0 || M==0 ) {
    return -1; /* must be non-negative numbers */
  }
  lkh->w = w;
  lkh->M = M;
  return 0;
}


/* choose parameters a and b for the hash function, set them accordingly. */
void lowl_key_hash_arm( lowl_key_hash* lkh ) { 
  /* choose parameters a and b */

  /* a must be an odd positive integer with a < 2^w. */
  lkh->a = (unsigned long) random();
  if ( lkh->a % 2 == 0 ) {
    lkh->a +=1;
  }
  if( lkh->a == 0 ) { /* make sure a isn't 0 */
    lkh->a = 1;
  }
  if ( 8*sizeof(lkh->a) > lkh->w ) {
    unsigned long long a_upperbound
      = (unsigned long long) powposint(2, lkh->w);
    lkh->a = (unsigned long) (lkh->a % a_upperbound);
  } 
  /* b must be a non-negative integer with b < 2^(w-M) */
  lkh->b = (unsigned long) (random() % powposint(2,lkh->w - lkh->M));
}

/* Motwani-Raghavan hash function. */

int motrag_hash_init( motrag_hash* lmh,
				unsigned int m, unsigned int n ) {
  unsigned int bigprime = (unsigned int) LOWLMATH_BIGPRIME;
  if( m > bigprime ) {
    return -1; // can't deal with an input universe this big.
  }
  lmh->m = m;
  lmh->n = n;
  /* select an appropriate prime p >= m. */
  // first figure out the smallest power powof2 such that m <= 2^powof2.
  unsigned int pow = 0;
  unsigned int testme = 1; // invariant: = 2^pow.
  while( testme < m ) {
    pow++;
    testme *= 2;
  }
  if( pow >= 32 ) { // m is extremely big. Really, we never want m this big.
    lmh->p = bigprime;
  } else {
    //lmh->p = lowlmath_usefulprimes[pow];
    lmh->p = get_useful_prime(pow);
  }
  return 0;
}

unsigned int motrag_map( unsigned int input, motrag_hash* lmh ) {
  unsigned int f = (lmh->a*input + lmh->b) % lmh->p;
  return f % lmh->n;
}

void motrag_hash_arm( motrag_hash* lmh ) {
  /* choose the parameters a and b for the hash function. */
  unsigned int r1 = (unsigned int) random();
  unsigned int r2 = (unsigned int) random();
  lmh->a = r1 % lmh->p;
  lmh->b = r2 % lmh->p;
}

/********************************************************
 *                                                      *
 *	Resizable array for use in hash tables.		*
 *                                                      *
 ********************************************************/

rarr_entry rarr_entry_from_kvpair( lowl_key k, lowl_count v) {
  rarr_entry result;
  result.key = k;
  result.value = v;
  return result;
}

int rarr_init(rarr* lr, unsigned int cap) {
  /* initialize a new resizable array, with given capacity.
        Return 0 if successful.
        Return -1 if there was a failure (namely, failure to allocate mem.) */
  lr->capacity = cap;
  lr->array = malloc( cap*sizeof(rarr_entry) );
  if( lr->array == NULL ) {
    return -1;
  }
  /* initialize all entries to be 0. */
  memset( lr->array, 0, cap*sizeof(rarr_entry) );
  return 0;
}

void rarr_clear(rarr* lr) {
  /* set the whole array to 0. */
  memset( lr->array, 0, lr->capacity*sizeof(rarr_entry) );
}

int rarr_set(rarr* lr, unsigned int loc, rarr_entry entry) {
  /* insert the given element into the resizable array at the given location.
        Return 0 if successful.
        Return -1 if location is out of range.  */
  if( loc >= lr->capacity ) {
    return -1;
  } else {
    /* for now, we're assuming that the entire array is in one contiguous
        piece of memory. In the future, it will probably make sense to
        change this so that the array is stored in several different
        not necessarily contiguous blocks of memory. This will be better for
        memory management (less opportunity for failure of malloc), but slightly
        worse for access speed and marginally more inconvenient in terms of
        bookkeeping.    */
    (lr->array)[loc] = entry;
  }
  return 0;
}

/* retrieve the element at the given location and copy its contents to
	the given address.	*/
int rarr_get(rarr* lr, unsigned int loc, rarr_entry* entry) {
  if( loc >= lr->capacity ) {
    return -1;
  } else {
    /* again, making the same contiguous memory assumption we made above. */
    *entry = (lr->array)[loc];
    return 0;
  }
}

int rarr_upsize(rarr* lr) {
  /* embiggen the array. */

  unsigned int oldCap = lr->capacity;
  unsigned int newCap = 2*oldCap;

  /* allocate the new memory. */
  rarr_entry* newarray = malloc( newCap*sizeof(rarr_entry) );
  if ( newarray == NULL ) {
    return -1;
  }

  /* copy the old array into the new one. */
  memcpy( newarray, lr->array, oldCap*sizeof(rarr_entry) );
  /* free the old memory, which we no longer need. */
  free( lr->array );
  /* clear the remaining new memory to be all "NULL" keys. */
  memset( &( newarray[oldCap] ), 0, oldCap*sizeof(rarr_entry) );
  /* set the array to point to the new memory. */
  lr->array = newarray;
  /* update the capacity of the array. */
  lr->capacity = newCap;

  return 0;
}

int rarr_downsize(rarr* lr) {
  /* disembiggen the array. */

  unsigned int oldCap = lr->capacity;
  unsigned int newCap = oldCap/2;

  /* allocate the new memory and verify malloc success. */
  rarr_entry* newarray = malloc( newCap*sizeof(rarr_entry) );
  if ( newarray == NULL ) {
    return -1;
  }

  /* copy the old array into the new one. We need only copy the
        first newCap elements of the array, since the fact that we are
        downsizing the array indicates that we do not need these entries. */
  memcpy( newarray, lr->array, newCap*sizeof(rarr_entry) );
  /* free the old memory, which we no longer need. */
  free( lr->array );
  /* set the array to point to the new memory. */
  lr->array = newarray;
  /* update the capacity of the array. */
  lr->capacity = newCap;

  return 0;
}

int rarr_destroy(rarr* lr) {
  /* deal with the various freeing of memory that needs to be done
        internal to the resizable array. */
  free( lr->array );
  lr->array = NULL;
  lr->capacity = 0;
  return 0;
}

/********************************************************
 *                                                      *
 *      Hash table for <lowl_key,lowl_count>            *
 *                                                      *
 ********************************************************/

/* hash table mapping lowl_keys to counts.
	We use cuckoo hashing, because it has nice properties and isn't
	terribly hard to implement. Refer to
	http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.25.4189	*/

int ht_key_to_count_init( ht_key_to_count* ht, unsigned int capacity ) {
  /* initialize a hash table mapping lowl_keys to counts.
	capacity is the desired size of the hash table,
	and must be a power of 2.
	If capacity isn't a power of 2, determine_logcardinality
	implicitly makes this the case. */
  //unsigned int M = determine_logcardinality(capacity);
  unsigned int M = (unsigned int) log2(capacity);
  if( M <= 0 ) {
    M = 1;
  }
  ht->hashfn = malloc( sizeof(lowl_key_hash) );
  ht->table = malloc( sizeof( rarr ) );

  unsigned int nbits_for_bitvector = powposint(2,M);
  ht->populace_table = malloc( sizeof( bitvector ) );
  bitvector_init( ht->populace_table, nbits_for_bitvector );
  bitvector_clear( ht->populace_table, nbits_for_bitvector);
 
  /* verify that all mallocs were successful before we move on. */
  if( ht->hashfn==NULL || ht->table==NULL || ht->populace_table==NULL ) {
    return LOWLERR_BADMALLOC;
  }

  lowl_key_hash_init( ht->hashfn, 8*sizeof(lowl_key), M );
  lowl_key_hash_arm( ht->hashfn );

  /* rarr_init sets all entries in the table to 0
	Capacity of the table is 2^M which is >= the given capacity */
  rarr_init( ht->table, powposint(2, M) );

  /* hash table is initially empty. */
  ht->size = 0;
  return 0;
}

float ht_key_to_count_loadfactor( ht_key_to_count* ht ) {
  return ht_key_to_count_size(ht) / ( (float) rarr_capacity(ht->table) );
}


int ht_key_to_count_findslot( ht_key_to_count* ht, lowl_key lkey,
				lowl_hashoutput* location) {
  /* Figure out if the given key is in this hash table.
	If it is, make the contents of *location hold the number of the
	slot in the table where lkey is stored.
	If it isn't, make *location the number of the first empty slot
	found by the hash/probe of lkey.
	Return a success code which tells us whether or not the key was
	found in the table.	*/ 
  
  lowl_hashoutput hout = multip_add_shift( lkey, ht->hashfn );

 /* this placeholder is for having a gander at what's currently stored
	in some given slot of the array.	*/
  rarr_entry placeholder_entry;
  rarr_get( ht->table, hout, &placeholder_entry );

  /* hash until we either find the key we're looking for OR until we
	find an empty spot in the array. */
  while(  ht_key_to_count_entryispopulated(ht, hout)  ) { 
    if( (lowl_key) placeholder_entry.key == lkey ) {
      /* found our key (or an empty slot). Store the location. */
      *location = hout;
      return LOWLHASH_INTABLE;
    } else {
      /* We found a key in this entry, but
	this isn't the key we're looking for.
	Linear probe to the next entry. */
      hout = (hout + 7) % rarr_capacity(ht->table);
      rarr_get( ht->table, hout, &placeholder_entry );
    }
  }
  /* left the while loop without finding our key, so hout is now pointing at
	an empty location in the table.	*/
  *location = hout;
  return LOWLHASH_NOTINTABLE;
}

int ht_key_to_count_set( ht_key_to_count* ht, lowl_key lkey, lowl_count val) {
  /* store (key,val) in the hash table. */

  /* If adding this entry will make the load factor too high,
        resize the hash table first and THEN insert.    */
  if( ht_key_to_count_loadfactor( ht ) >= HT_KEY_TO_COUNT_MAXLOAD ) {
    ht_key_to_count_upsize( ht );
    return ht_key_to_count_set( ht, lkey, val);
  }

  /* If the given key wasn't in the hash table. We need to add it.
        The point of ht_key_to_count_find is that when the function
        returns, location is either a place where the key was located,
        or points to the first empty slot that the probe found if the
        key wasn't in the table.
	The success code tells us whether we found the key in the table
	or not.	*/

  lowl_hashoutput location;
  int is_in_table = ht_key_to_count_findslot( ht, lkey, &location );
  /* write our entry into this location. */
  rarr_entry entry_to_store = rarr_entry_from_kvpair(lkey, val);
  rarr_set( ht->table, location, entry_to_store );
  ht->size++;
  return is_in_table;
}

int ht_key_to_count_get( ht_key_to_count* ht, lowl_key lkey, lowl_count* val) {
  /* retrieve the value stored for the given key. Place that value in the
	address pointer to by lowl_count* val.	*/

  lowl_hashoutput location;
  int is_in_table = ht_key_to_count_findslot( ht, lkey, &location );
  if( is_in_table==LOWLHASH_NOTINTABLE ) {
    /* given key wasn't in the hash table. */
    return LOWLHASH_NOTINTABLE;
  } else {
    /* given key was in the hash table.
	Copy the value to val and return successful exit code. */
    rarr_entry entry;
    rarr_get(ht->table, location, &entry);
    *val = entry.value;
    return LOWLHASH_INTABLE;
  } 
}

int ht_key_to_count_entryispopulated( ht_key_to_count* ht,
                                        lowl_hashoutput location) {
  /* return 1 if an entry exists in the hash table at the given location.
	return 0 otherwise. */
  return bitvector_lookup( ht->populace_table, location);
}

int ht_key_to_count_upsize( ht_key_to_count* ht ) {
  /* resize the hash table.
	This involves a number of steps:
	1) Create a new array of twice the size of the existing one.
	2) Create a new populace_table of the same size as this new array.
	3) Hash all old elements from the old array into the new one.
		update the new populace table in so doing.
	4) Free the old, now unused, memory.	*/
  if( ht == NULL ) {
    return LOWLERR_BADINPUT;
  } else {
    return LOWLERR_BADMALLOC;
  }
}
 
void ht_key_to_count_clear( ht_key_to_count* ht ) {
  rarr_clear( ht->table );
  ht->size = 0;
} 

void ht_key_to_count_destroy( ht_key_to_count* ht ) {
  free( ht->hashfn );
  ht->hashfn = NULL;
  rarr_destroy( ht->table );
  free( ht->table );
  ht->table = NULL;
  free( ht->populace_table );
  ht->populace_table = NULL;
  ht->size = 0;
}


/********************************************************
 *                                                      *
 *      Hash table for storing keys                     *
 *                                                      *
 ********************************************************/

// Not actually sure whether or not this will be necessary, yet.
