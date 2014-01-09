#include <stdlib.h>
#include <string.h>
#include "lowl_types.h"
#include "lowl_hash.h"

/********************************************************
 *                                                      *
 *      Begin temporarily-deprecated code.              *
 *                                                      *
 ********************************************************/

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

/********************************************************
 *                                                      *
 *      End temporarily-deprecated code.                *
 *                                                      *
 ********************************************************/


/********************************************************
 *                                                      *
 *      Hash functions.					*
 *                                                      *
 ********************************************************/

/* multiply-add-shift hash function. */
lowl_hashoutput multip_add_shift(lowl_key x, lowl_key_hash* h) {
  return (lowl_hashoutput) ((h->a)*x + (h->b)) >> (h->w - h->M);
}


/********************************************************
 *                                                      *
 *	Resizable array for use in hash tables.		*
 *                                                      *
 ********************************************************/

int lowl_rarr_init(lowl_rarr* lr, unsigned int cap) {
  /* initialize a new resizable array, with given capacity.
        Return 0 if successful.
        Return -1 if there was a failure (namely, failure to allocate mem.) */
  lr->capacity = cap;
  lr->size = 0; /* initially empty */
  lr->array = malloc( cap*sizeof(lowl_count) );
  if( lr->array == NULL ) {
    return -1;
  }
  /* initialize all entries to be 0. */
  memset( lr->array, 0, cap*sizeof(lowl_count) );
  return 0;
}

int lowl_rarr_insert(lowl_rarr* lr, unsigned int loc, lowl_count elmt) {
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

int lowl_rarr_upsize(lowl_rarr* lr) {
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

int lowl_rarr_downsize(lowl_rarr* lr) {
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

int lowl_rarr_destroy(lowl_rarr* lr) {
  /* deal with the various freeing of memory that needs to be done
        internal to the resizable array. */
  free( lr->array );
  lr->size = 0;
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
