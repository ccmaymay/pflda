#include <stdlib.h>
#include <string.h>
#include "lowl_types.h"
#include "lowl_math.h"
#include "lowl_hash.h"

#define CUCKOO_NHASHES 8
#define CUCKOO_REHASH -2
#define LOWLHASH_INTABLE 0
#define LOWLHASH_NOTINTABLE 10

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
lowl_hashoutput mod_fnv(const char *data, size_t len, unsigned int salt) {
  unsigned int p = 16777619;
  lowl_hashoutput hash = 1315423911;
  hash = (hash ^ salt) * p;
  for (size_t i = 0; i < len; ++i)
    hash = (hash ^ data[i]) * p;
  hash += hash << 13;
  hash ^= hash >> 7;
  hash += hash << 3;
  hash ^= hash >> 17;
  hash += hash << 5;
  return hash;
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
int rarr_get(rarr* lr, unsigned int loc, lowl_count* entry) {
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
  lowl_count* newarray = malloc( newCap*sizeof(rarr_entry) );
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
  lowl_count* newarray = malloc( newCap*sizeof(rarr_entry) );
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

typedef struct ht_key_to_count{
  /* we will actually use one hash function to get CUCKOO_NHASHES
	hash functions using the same trick we use in the bloom filter.
	h_i(x) = f(x) + i*g(x) */
  lowl_key_hash* hashfn;
  rarr* table;
  unsigned int size; /* number of elements we've inserted. */
}ht_key_to_count;

int ht_key_to_count_init( ht_key_to_count* ht, unsigned int capacity ) {
  /* initialize a hash table mapping lowl_keys to counts.
	capacity is the desired size of the hash table,
	and must be a power of 2.
	If capacity isn't a power of 2, determine_logcardinality
	implicitly makes this the case. */
  unsigned int M = determine_logcardinality(capacity);
  ht->hashfn = malloc( sizeof(lowl_key_hash) );
  ht->table = malloc( sizeof( rarr ) );
  if( ht->hashfn==NULL || ht->table==NULL ) {
    return -1;
  }

  lowl_key_hash_init( ht->hashfn, 8*sizeof(lowl_key), M );
  lowl_key_hash_arm( ht->hashfn );

  /* rarr_init sets all entries in the table to 0 */
  rarr_init( ht->table, powposint(2, M) );

  /* hash table is initially empty. */
  ht->size = 0;
}

int ht_key_to_count_set( ht_key_to_count* ht, lowl_key lkey, lowl_count val) {
  /* store (key,val) in the hash table. */
  lowl_hashoutput hout = multip_add_shift( lkey, ht->hashfn );

  rarr_entry placeholder_entry, entry_to_insert;

  /* we need to find a place in the hash table for this entry. */
  entry_to_insert = rarr_entry_from_kvpair(lkey, val);

 /* the placeholder is to let us have a gander at what's currently stored
	in some given slot of the array.	*/
  rarr_get( ht->table, hout, &placeholder_entry );

  /* hash until we either find the key we're looking for OR until we
	find an empty spot in the array. */
  while( (lowl_key) entry.key != 0 ) { 
    if( (lowl_key) entry.key == lkey || (lowl_key) entry.key == 0 ) {
      rarr_set( ht->table, hout, rarr_entry_from_kvpair(lkey, val) );
      return LOWLHASH_INTABLE;
    } else {
      /* linear probe. */
      hout = (hout + 7) % rarr_size(ht->table);
      rarr_get( ht->table, hout, &placeholder_entry );
    }
  }
  (rarr->array)[hout] = make_rarr_entry_from_kvpair(lkey, val);
  return LOWLHASH_NOTINTABLE;
} 

int ht_key_to_count_get( ht_key_to_count* ht, lowl_key lkey, lowl_count* val) {
  lowl_hashoutput hout = multip_add_shift( lkey, ht->hashfn );
  
  rarr_entry entry;
  rarr_get( ht->table, hout, &entry );
  while( (lowl_key) entry.key != 0 ) {
    if( (lowl_key) entry.key == lkey ) {
      *val = entry.value;
      return LOWLHASH_INTABLE;
    } else {
      /* linear probe. */
      hout = (hout + 7) % rarr_size(ht->table);
      rarr_get( ht->table, hout, &entry );
    }
  }
  return LOWLHASH_NOTINTABLE;
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
  ht->size = 0;
}


/********************************************************
 *                                                      *
 *      Hash table for storing keys                     *
 *                                                      *
 ********************************************************/

// Not actually sure whether or not this will be necessary, yet.
