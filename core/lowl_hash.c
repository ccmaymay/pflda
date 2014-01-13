#include <stdlib.h>
#include <string.h>
#include "lowl_types.h"
#include "lowl_math.h"
#include "lowl_hash.h"

/********************************************************
 *                                                      *
 *      Hash functions.					*
 *                                                      *
 ********************************************************/

/* multiply-add-shift hash function. */
lowl_hashoutput multip_add_shift(lowl_key x, lowl_key_hash* h) {
  return ((lowl_hashoutput)
	 ((h->a)*((unsigned int)x) + (h->b))) >> (h->w - h->M) ;
}

int lowl_key_hash_init( lowl_key_hash* lkh, unsigned int w, unsigned int M) {
  if( w==0 || M==0 ) return -1; /* must be non-negative numbers */
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
  lr->array = malloc( cap*sizeof(lowl_count) );
  if( lr->array == NULL ) {
    return -1;
  }
  /* initialize all entries to be 0. */
  memset( lr->array, 0, cap*sizeof(lowl_count) );
  return 0;
}

int rarr_set(rarr* lr, unsigned int loc, lowl_count elmt) {
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
    (lr->array)[loc] = elmt;
  }
  return 0;
}

/* retrieve the element at the given location and copy its contents to
	the given address.	*/
int rarr_get(rarr* lr, unsigned int loc, lowl_count* elmt) {
  if( loc >= lr->capacity ) {
    return -1;
  } else {
    /* again, making the same contiguous memory assumption we made above. */
    *elmt = (lr->array)[loc];
    return 0;
  }
}

int rarr_upsize(rarr* lr) {
  /* embiggen the array. */

  unsigned int oldCap = lr->capacity;
  unsigned int newCap = 2*oldCap;

  /* allocate the new memory. */
  lowl_count* newarray = malloc( newCap*sizeof(lowl_count) );
  if ( newarray == NULL ) {
    return -1;
  }

  /* copy the old array into the new one. */
  memcpy( newarray, lr->array, oldCap*sizeof(lowl_count) );
  /* free the old memory, which we no longer need. */
  free( lr->array );
  /* clear the remaining new memory to be all "NULL" keys. */
  memset( &( newarray[oldCap] ), 0, oldCap*sizeof(lowl_count) );
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
  lowl_count* newarray = malloc( newCap*sizeof(lowl_count) );
  if ( newarray == NULL ) {
    return -1;
  }

  /* copy the old array into the new one. We need only copy the
        first newCap elements of the array, since the fact that we are
        downsizing the array indicates that we do not need these entries. */
  memcpy( newarray, lr->array, newCap*sizeof(lowl_count) );
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
  lr->capacity = 0;
  return 0;
}

/********************************************************
 *                                                      *
 *      Hash table for storing counts                   *
 *                                                      *
 ********************************************************/



/********************************************************
 *                                                      *
 *      Hash table for storing keys                     *
 *                                                      *
 ********************************************************/

// Not actually sure whether or not this will be necessary, yet.
