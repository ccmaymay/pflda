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
 *      Hash table for storing keys                     *
 *                                                      *
 ********************************************************/

// Not actually sure whether or not this will be necessary, yet.
