#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "lowl_types.h"
#include "lowl_hash.h"
#include "lowl_math.h"

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
  unsigned int *bins = malloc( numbins*sizeof(unsigned int) );
  
  if ( bins == NULL ) {
    fprintf(stderr, "Memory allocation failed while setting up bins for chi2 test for hash functions.\n");
    exit( EXIT_FAILURE );
  }
  
  int numtrials = 100; /* number of hash function trials */
  int numints = 4096;
  
  /* allocate a Lowl_Int_Hash and set its w and M parameters, which do
  	not change from trial to trial. */
  Lowl_Int_Hash* lihash = malloc(sizeof( Lowl_Int_Hash) );
  
  lihash->M = (unsigned int) log2(numbins); // e.g., 2^6=64, the # of bins.
  /* we're hashing unsigned ints.
  	the w parameter to the multiply-add-shift is the number of bits
  	needed to represent the objects that we are hashing, so
  	we need w to be the number of bits in an unsigned int. */
  lihash->w = (unsigned int) 8*sizeof(unsigned int);

  /* we will tabulate chi^2 statistics for the trials. */
  double* chi2scores = malloc(numtrials*sizeof(double));
  if ( chi2scores == NULL ) {
    fprintf(stderr, "Memory allocation failed while setting up score array for chi2 test for hash functions.\n");
    exit( EXIT_FAILURE );
  }
  
  /* seed and begin trials. We reset the a and b parameters of the hash function
  	with each trial. */
  srandom(3355);
  int n,j;
  unsigned int current_int;
  unsigned int current_hash;
  /* for calculating chi2 scores. */
  double expected = ((double)numints) / ((double)numbins);
  double observed, diff;
  for( n=0; n<numtrials; n++ ) {
    /* reset the counts. */
    memset( bins, 0, numbins*sizeof(unsigned int) );

    /* choose parameters a and b */
    /* a must be an odd positive integer. */
    lihash->a = (unsigned long) random();
    if ( lihash->a % 2 == 0 ) { 
      lihash->a +=1;
    }
    /* a,b the parameters of the hash function, must both be less than 2^w */
    if ( 8*sizeof(lihash->a) > lihash->w ||
		8*sizeof(lihash->b) > lihash->w ) {
      unsigned long long ab_upperbound
        = (unsigned long long) lowlmath_powposint(2, lihash->w);
      lihash->a = lihash->a % ab_upperbound;
      lihash->b = (unsigned long) random() % ab_upperbound;
    }
    for( j=0; j<numints; j++ ) {
      current_int = (unsigned int) random();
      current_hash = multip_add_shift(current_int, lihash);
      bins[current_hash] += 1;
    }

    /* calculate chi2 statistic. */
    chi2scores[n] = 0.0;
    for( j=0; j<numbins; j++) {
      observed = (double) bins[j];
      diff = observed - expected;
      chi2scores[n] += pow(diff, 2.0)/expected;
    }
  }
  /* "interpret" the chi2 scores. */
  /* critical values retrieved from
	http://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm */
  double cval_upper = 82.529; // critical value for 63 df at alpha=0.95
  double cval_lower = 45.741; // critical value for 63 df at alpha=0.95
  double cval_twotailed_upper = 86.830;
  double cval_twotailed_lower = 42.950;
  int numfailed_upper = 0;
  int numfailed_lower = 0;
  int numfailed_twotailed = 0;
  printf("Chi2 test of uniformity of hash function output:\n");
  for( n=0; n<numtrials; n++ ) {
    if( chi2scores[n] >= cval_upper ) {
      numfailed_upper++;
    }
    if( chi2scores[n] <= cval_lower ) {
      numfailed_lower++;
    }
    if( chi2scores[n] >= cval_twotailed_upper
		|| chi2scores[n] <= cval_twotailed_lower ) {
      numfailed_twotailed++;
    }
  }
  printf("%d of %d trials failed the upper-tailed chi2 test at alpha=0.05.\n",
		numfailed_upper, numtrials );
  printf("%d of %d trials failed the lower-tailed chi2 test at alpha=0.05.\n",
		numfailed_lower, numtrials );
  printf("%d of %d trials failed the two-tailed chi2 test at alpha=0.05.\n",
		numfailed_twotailed, numtrials );
  printf("\n");

  printf("All tests completed.\n");
  return 0;
}
