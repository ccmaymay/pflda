#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "lowl_types.h"
#include "lowl_hash.h"
#include "lowl_math.h"

void run_chi2( double* scores, int numtrials, lowl_key_hash* lkh,
		unsigned int* bins, int numbins,
		lowl_key* keys, int numkeys ) {
  /* run a chi2 test using the given hash structure.
	double* scores will hold int numtrials scores.
	Each score is obtained by hashing int numkeys lowl_keys,
		given in lowl_key*, into unsigned int* bins, where we
		will accumulate counts. */ 
  int n,j;
  lowl_key current_key;
  lowl_hashoutput current_hash;
  /* for calculating chi2 scores. */
  double expected = ((double)numkeys) / ((double)numbins);
  double observed, diff;
  for( n=0; n<numtrials; n++ ) {
    /* reset the counts. */
    memset( bins, 0, numbins*sizeof(unsigned int) );

    /* choose parameters a and b */
    /* a must be an odd positive integer. */
    lkh->a = (unsigned long) random();
    if ( lkh->a % 2 == 0 ) { 
      lkh->a +=1;
    }
    /* a,b the parameters of the hash function, must both be less than 2^w */
    if ( 8*sizeof(lkh->a) > lkh->w ||
		8*sizeof(lkh->b) > lkh->w ) {
      unsigned long long ab_upperbound
        = (unsigned long long) lowlmath_powposint(2, lkh->w);
      lkh->a = lkh->a % ab_upperbound;
      lkh->b = (unsigned long) random() % ab_upperbound;
    }
    /* hash the keys. */
    for( j=0; j<numkeys; j++ ) {
      current_key = keys[j];
      current_hash = multip_add_shift(current_key, lkh);
      bins[current_hash] += 1;
    }

    /* calculate chi2 statistic. */
    scores[n] = 0.0;
    for( j=0; j<numbins; j++) {
      observed = (double) bins[j];
      diff = observed - expected;
      scores[n] += pow(diff, 2.0)/expected;
    }
  }
  return;
}

void interp_chi2( double *chi2scores, int numtrials ) {
  /* critical values retrieved from
        http://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm */
  double cval_upper = 82.529; // critical value for 63 df at alpha=0.95
  double cval_lower = 45.741; // critical value for 63 df at alpha=0.95
  double cval_twotailed_upper = 86.830;
  double cval_twotailed_lower = 42.950;

  int failed_upper,failed_lower,failed_twotailed;
  failed_upper=failed_lower=failed_twotailed=0;

  int n;
  for( n=0; n<numtrials; n++ ) {
    if( chi2scores[n] >= cval_upper ) {
      failed_upper++;
    }
    if( chi2scores[n] <= cval_lower ) {
      failed_lower++;
    }
    if( chi2scores[n] >= cval_twotailed_upper
                || chi2scores[n] <= cval_twotailed_lower ) {
      failed_twotailed++;
    }
  }
  printf("%d of %d trials failed the upper-tailed chi2 test at alpha=0.05.\n",
                failed_upper, numtrials );
  printf("%d of %d trials failed the lower-tailed chi2 test at alpha=0.05.\n",
                failed_lower, numtrials );
  printf("%d of %d trials failed the two-tailed chi2 test at alpha=0.05.\n",
                failed_twotailed, numtrials );
  printf("\n"); 
}

int main( int argc, char **argv ) {

  /**************************************************************
   *								*
   *	 Tests for lowl_math.c 					*
   *								*
   **************************************************************/
  assert( lowlmath_powposint(2, 0) == 1);
  assert( lowlmath_powposint(2, 1) == 2);
  assert( lowlmath_powposint(2, 10) == 1024);
  assert( lowlmath_powposint(2, 11) == 2048);
  assert( lowlmath_powposint(2, 31) == 2147483648);

  /**************************************************************
   *								*
   *	 Tests for lowl_hash.c 					*
   *								*
   **************************************************************/

  /* We would like to verify that the multiply-add-shift hash function
  	results in reasonably well-distribute behavior when hashing over
  	a large array of possible outcomes.
  
     We will hash unsigned ints (>=32 bits, platform-dependent)
  	to a set of 64 possible bins. */
  
  int numbins = 64;
  int numtrials = 100; /* number of hash function trials */
  int numkeys = 4096;

  /* we store all keys to be hashed during a trial in a signle array. */
  lowl_key* keys = malloc( numkeys*sizeof(lowl_key) );
  
  /* allocate a lowl_key_hash and set its w and M parameters, which do
  	not change from trial to trial. */
  lowl_key_hash* lkhash = malloc(sizeof( lowl_key_hash) );
  
  lkhash->M = (unsigned int) log2(numbins); // e.g., 2^6=64, the # of bins.
  /* we're hashing unsigned ints.
  	the w parameter to the multiply-add-shift is the number of bits
  	needed to represent the objects that we are hashing, so
  	we need w to be the number of bits in an unsigned int. */
  lkhash->w = (unsigned int) 8*sizeof(unsigned int);

  /* we will tabulate chi^2 statistics for the trials. */
  double* chi2scores = malloc(numtrials*sizeof(double));
  /* we accumulate counts in unsigned int* bins. */
  unsigned int *bins = malloc( numbins*sizeof(unsigned int) );

  if ( keys==NULL || lkhash==NULL || chi2scores==NULL || bins==NULL ) {
    fprintf(stderr, "Memory allocation failed while setting up chi2 test.\n");
    exit( EXIT_FAILURE );
  }
  
  /* seed and begin trials. We reset the a and b parameters of the hash function
  	with each trial. */
  srandom(3355);

  /* first test will just be hashing a random set of integers. */
  // populate the integer list.
  int i;
  for( i=0; i < numkeys; i++) {
    keys[i] = (lowl_key) random();
  }
  printf("Chi2 test of uniformity of hash function output when inputs are randomly-drawn unsigned ints:\n");
  run_chi2( chi2scores, numtrials, lkhash, bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  /* Now, we will hash sequential keys and test again. */
  for( i=0; i<numkeys; i++) {
    keys[i] = (lowl_key) i;
  }
  printf("Chi2 test of uniformity of hash function output when inputs are sequential keys starting at 0:\n");
  run_chi2( chi2scores, numtrials, lkhash, bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  /* Now, we will hash sequential keys starting from a non-zero number. */
  int offset = 49327;
  for( i=0; i<numkeys; i++) {
    keys[i] = (lowl_key) i + offset; 
  }
  printf("Chi2 test of uniformity of hash function output when inputs are sequential keys starting at %d:\n", offset);
  run_chi2( chi2scores, numtrials, lkhash, bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  /* Keys that are sequential by 2s. */
  for( i=0; i<numkeys; i++) {
    keys[i] = (lowl_key) 2*i; 
  }
  printf("Chi2 test of uniformity of hash function output when inputs are sequential by 2s, starting at 0:\n");
  run_chi2( chi2scores, numtrials, lkhash, bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  /* free the memory that we no longer need. */
  free( chi2scores );
  free( bins );
  free( keys );
  free( lkhash );

  /******************************************************
   *							*
   *	Tests for resizable arrays.			*
   *							*
   ******************************************************/

  printf("Tests for resizable arrays.\n");

  lowl_rarr *lr;
  lr = malloc( sizeof(lowl_rarr) );
  if( lr == NULL ) {
    printf( "Memory allocation failed in testing resizable array.\n");
  }
  unsigned int orig_cap = 16;
  /* initialize a resizable array with 16 slots. */
  lowl_rarr_init(lr, orig_cap);
  assert( lr->capacity == orig_cap );
  /* verify that entries of lr are zero, as they ought to be */
  lowl_count contents;
  int succ;
  for( i=0; i < lr->capacity; i++ ) {
    succ = lowl_rarr_get(lr, (unsigned int) i, &contents);
    assert( contents==0 ); 
    assert( succ==0 );
  }
  /* set some entries to be non-zero and verify that this works correctly. */
  lowl_count testcount1 = 1969;
  lowl_count testcount2 = 42;
  succ = lowl_rarr_set(lr, 5, testcount1);
  assert( succ == 0 );
  succ = lowl_rarr_set(lr, 10, testcount2);
  assert( succ == 0 );
  succ = lowl_rarr_get(lr, 5, &contents);
  assert( succ==0 );
  assert( contents == testcount1 );
  succ = lowl_rarr_get(lr, 10, &contents);
  assert( succ==0 );  
  assert( contents == testcount2 );
  /* try setting an element that is currently out of range. */
  lowl_count testcount3 = 123456;
  succ = lowl_rarr_set(lr, 16, testcount3);
  assert( succ != 0 );

  /* upsize the array. */
  lowl_rarr_upsize( lr );
  assert( lr->capacity == 2*orig_cap );
  /* check that newly created memory is zero'd */
  for( i=orig_cap; i < lr->capacity; i++ ) {
    succ = lowl_rarr_get(lr, (unsigned int) i, &contents);
    assert( contents==0 );
    assert( succ==0 );
  }
  /* now try inserting the same element. */
  succ = lowl_rarr_set(lr, 16, testcount3);
  assert( succ == 0 );
  succ = lowl_rarr_get(lr, 16, &contents);
  assert( succ==0 );
  assert( contents == testcount3 );
  /* verify that previous elements were preserved correctly. */
  succ = lowl_rarr_get(lr, 5, &contents);
  assert( succ==0 );
  assert( contents == testcount1 );
  succ = lowl_rarr_get(lr, 10, &contents);
  assert( succ==0 );
  assert( contents == testcount2 );


  /* downsize the array. */
  succ = lowl_rarr_downsize( lr );
  assert( lr->capacity == orig_cap );

  lowl_rarr_destroy( lr );
  free( lr );
 



  printf("All tests completed.\n");
  return 0;
}
