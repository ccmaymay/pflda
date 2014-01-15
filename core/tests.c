#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "lowl_types.h"
#include "lowl_hash.h"
#include "lowl_math.h"
#include "lowl_sketch.h"

#define test(name, f) do { printf("Testing %s... ", name); int ret; if ((ret = (f))) printf("error %d\n", ret); else printf("ok\n"); } while (0);
#define test_bool(name, f) test(name, (f ? 0 : 1));

void interp_chi2( double* scores, int numtrials);

void run_chi2_lkh( double* scores, int numtrials, lowl_key_hash* lkh,
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
    lowl_key_hash_arm( lkh );

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

void run_chi2_motrag( double* scores, int numtrials, motrag_hash* motwani,
                unsigned int* bins, int numbins,
                unsigned int* keys, int numkeys ) {
  /* run a chi2 test using the given hash structure.
        double* scores will hold int numtrials scores.
        Each score is obtained by hashing int numkeys unsigned ints,
                into unsigned int* bins, where we will accumulate counts. */
  int n,j;
  unsigned int current_key;
  unsigned int current_hash;
  /* for calculating chi2 scores. */
  double expected = ((double)numkeys) / ((double)numbins);
  double observed, diff;
  for( n=0; n<numtrials; n++ ) {
    /* reset the counts. */
    memset( bins, 0, numbins*sizeof(unsigned int) );

    /* choose parameters a and b */
    motrag_hash_arm( motwani );

    /* hash the keys. */
    for( j=0; j<numkeys; j++ ) {
      current_key = keys[j];
      current_hash = motrag_map(current_key, motwani);
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

void run_multip_add_shift_tests( void ) {
  printf("=== Chi2 tests of multiply-add-shift hash function. ===\n");

  /* allocate the necessary memory for running tests. */
  int numbins = 64;
  int numtrials = 100; /* number of hash function trials */
  int numkeys = 4096;

  /* we store all keys to be hashed during a trial in a signle array. */
  lowl_key* keys = malloc( numkeys*sizeof(lowl_key) );

  /* allocate a lowl_key_hash and set its w and M parameters, which do
        not change from trial to trial. */
  lowl_key_hash* lkhash = malloc(sizeof( lowl_key_hash) );

  unsigned int M = (unsigned int) log2(numbins); // e.g., 2^6=64, the # of bins
  /* we're hashing unsigned ints.
        the w parameter to the multiply-add-shift is the number of bits
        needed to represent the objects that we are hashing, so
        we need w to be the number of bits in a lowl_key. */
  unsigned int w = (unsigned int) 8*sizeof(lowl_key);

  lowl_key_hash_init( lkhash, w, M);

  /* we will tabulate chi^2 statistics for the trials. */
  double* chi2scores = malloc(numtrials*sizeof(double));
  /* we accumulate counts in unsigned int* bins. */
  unsigned int *bins = malloc( numbins*sizeof(unsigned int) );

  if ( keys==NULL || lkhash==NULL || chi2scores==NULL || bins==NULL ) {
    fprintf(stderr, "Memory allocation failed while setting up multiply-add-shift chi2 tests.\n");
    exit( EXIT_FAILURE );
  }

  /* first test will just be hashing a random set of integers. */
  // populate the integer list.
  int i;
  for( i=0; i < numkeys; i++) {
    keys[i] = (lowl_key) random();
  }
  printf("Chi2 test of uniformity of hash function output when inputs are randomly-drawn unsigned ints:\n");
  run_chi2_lkh( chi2scores, numtrials, lkhash, bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  /* Now, we will hash sequential keys and test again. */
  for( i=0; i<numkeys; i++) {
    keys[i] = (lowl_key) i;
  }
  printf("Chi2 test of uniformity of hash function output when inputs are sequential keys starting at 0:\n");
  run_chi2_lkh( chi2scores, numtrials, lkhash, bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  /* Now, we will hash sequential keys starting from a non-zero number. */
  int offset = 49327;
  for( i=0; i<numkeys; i++) {
    keys[i] = (lowl_key) i + offset;
  }
  printf("Chi2 test of uniformity of hash function output when inputs are sequential keys starting at %d:\n", offset);
  run_chi2_lkh( chi2scores, numtrials, lkhash, bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  /* Keys that are sequential by 2s. */
  for( i=0; i<numkeys; i++) {
    keys[i] = (lowl_key) 2*i;
  }
  printf("Chi2 test of uniformity of hash function output when inputs are sequential by 2s, starting at 0:\n");
  run_chi2_lkh( chi2scores, numtrials, lkhash, bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  /* Keys that are sequential by 2s. */
  for( i=0; i<numkeys; i++) {
    keys[i] = (lowl_key) 3*i;
  }
  printf("Chi2 test of uniformity of hash function output when inputs are sequential by 103s, starting at 0:\n");
  run_chi2_lkh( chi2scores, numtrials, lkhash, bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  free( chi2scores );
  free( lkhash );
  free( bins );
  free( keys);

  return;
}

void run_motwani_tests( void ) {
  printf("=== Running chi2 tests for Motwani-Raghavan hash function. ===\n");

  /* allocate the necessary memory for running tests. */
  int numbins = 64;
  int numtrials = 100; /* number of hash function trials */
  int numkeys = 4096;
  int m = 65536; /* size of the input universe that we're going to hash. */

  /* we store all keys to be hashed during a trial in a signle array. */
  unsigned int* keys = malloc( numkeys*sizeof(unsigned int) );

  /* allocate a lowl_key_hash and set its w and M parameters, which do
        not change from trial to trial. */
  motrag_hash* motwani = malloc(sizeof( motrag_hash) );

  motrag_hash_init( motwani, m, numbins );

  /* we will tabulate chi^2 statistics for the trials. */
  double* chi2scores = malloc(numtrials*sizeof(double));
  /* we accumulate counts in unsigned int* bins. */
  unsigned int *bins = malloc( numbins*sizeof(unsigned int) );

  if ( keys==NULL || motwani==NULL || chi2scores==NULL || bins==NULL ) {
    fprintf(stderr, "Memory allocation failed while setting up Motwani-Raghavan chi2 tests.\n");
    exit( EXIT_FAILURE );
  }

  /* first test will just be hashing a random set of integers. */
  // populate the integer list.
  int i;
  for( i=0; i < numkeys; i++) {
    keys[i] = (unsigned int) random();
    keys[i] = keys[i] % m;
  }
  printf("Chi2 test of uniformity of hash function output when inputs are randomly-drawn unsigned ints:\n");
  run_chi2_motrag(chi2scores, numtrials, motwani,
			bins, numbins, keys, numkeys);
  interp_chi2( chi2scores, numtrials );

  /* Now, we will hash sequential keys and test again. */
  for( i=0; i<numkeys; i++) {
    keys[i] = (unsigned int) i;
    keys[i] = keys[i] % m;
  }
  printf("Chi2 test of uniformity of hash function output when inputs are sequential keys starting at 0:\n");
  run_chi2_motrag(chi2scores, numtrials, motwani,
			bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  /* Now, we will hash sequential keys starting from a non-zero number. */
  int offset = 49327;
  for( i=0; i<numkeys; i++) {
    keys[i] = (unsigned int) i + offset;
    keys[i] = keys[i] % m;
  }
  printf("Chi2 test of uniformity of hash function output when inputs are sequential keys starting at %d:\n", offset);
  run_chi2_motrag(chi2scores, numtrials, motwani,
			bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  /* Keys that are sequential by 2s. */
  for( i=0; i<numkeys; i++) {
    keys[i] = (unsigned int) 2*i;
    keys[i] = keys[i] % m;
  }
  printf("Chi2 test of uniformity of hash function output when inputs are sequential by 2s, starting at 0:\n");
  run_chi2_motrag(chi2scores, numtrials, motwani,
			bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  /* Keys that are sequential by 2s. */
  for( i=0; i<numkeys; i++) {
    keys[i] = (unsigned int) 3*i;
    keys[i] = keys[i] % m;
  }
  printf("Chi2 test of uniformity of hash function output when inputs are sequential by 103s, starting at 0:\n");
  run_chi2_motrag(chi2scores, numtrials, motwani,
			bins, numbins, keys, numkeys );
  interp_chi2( chi2scores, numtrials );

  free( chi2scores );
  free( bins );
  free( keys );
  free( motwani );

  return;
}


void run_chi2_lmh( double* scores, int numtrials, motrag_hash* lmh,
		unsigned int* bins, int numbins,
		unsigned int* keys, int numkeys ) {
  /* run a chi2 test using the given hash structure.
	double* scores will hold int numtrials scores.
	Each score is obtained by hashing int numkeys lowl_keys,
		given in lowl_key*, into unsigned int* bins, where we
		will accumulate counts. */
  int n,j;
  unsigned int current_key;
  unsigned int current_hash;
  /* for calculating chi2 scores. */
  double expected = ((double)numkeys) / ((double)numbins);
  double observed, diff;
  for( n=0; n<numtrials; n++ ) {
    /* reset the counts. */
    memset( bins, 0, numbins*sizeof(unsigned int) );

    /* choose parameters a and b */
    motrag_hash_arm( lmh );

    /* hash the keys. */
    for( j=0; j<numkeys; j++ ) {
      current_key = keys[j];
      current_hash = motrag_map( current_key, lmh );
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

void run_resizablearray_tests() {
  printf("=== Running tests for resizable arrays. ===\n");

  rarr *lr;
  lr = malloc( sizeof(rarr) );
  if( lr == NULL ) {
    printf( "Memory allocation failed in testing resizable array.\n");
  }
  unsigned int orig_cap = 16;
  /* initialize a resizable array with 16 slots. */
  rarr_init(lr, orig_cap);
  assert( lr->capacity == orig_cap );
  /* verify that entries of lr are zero, as they ought to be */
  rarr_entry contents;
  int succ;
  unsigned int i;
  for( i=0; i < lr->capacity; i++ ) {
    succ = rarr_get(lr, (unsigned int) i, &contents);
    assert( contents.key==0 );
    assert( contents.value==0 );
    assert( succ==0 );
  }
  /* set some entries to be non-zero and verify that this works correctly. */
  lowl_count testcount1 = 1969;
  rarr_entry testentry1 = rarr_entry_from_kvpair(1,testcount1);
  lowl_count testcount2 = 42;
  rarr_entry testentry2 = rarr_entry_from_kvpair(2,testcount2);
  succ = rarr_set(lr, 5, testentry1);
  assert( succ == 0 );
  succ = rarr_set(lr, 10, testentry2);
  assert( succ == 0 );
  succ = rarr_get(lr, 5, &contents);
  assert( succ==0 );
  assert( contents.value == testcount1 );
  succ = rarr_get(lr, 10, &contents);
  assert( succ==0 );
  assert( contents.value == testcount2 );
  /* try setting an element that is currently out of range. */
  lowl_count testcount3 = 123456;
  rarr_entry testentry3 = rarr_entry_from_kvpair(3,testcount3);
  succ = rarr_set(lr, 16, testentry3);
  assert( succ != 0 );

  /* upsize the array. */
  rarr_upsize( lr );
  assert( lr->capacity == 2*orig_cap );
  /* check that newly created memory is zero'd */
  for( i=orig_cap; i < lr->capacity; i++ ) {
    succ = rarr_get(lr, (unsigned int) i, &contents);
    assert( contents.value==0 );
    assert( contents.key==0 );
    assert( succ==0 );
  }
  /* now try inserting the same element. */
  succ = rarr_set(lr, 16, testentry3);
  assert( succ == 0 );
  succ = rarr_get(lr, 16, &contents);
  assert( succ==0 );
  assert( contents.value == testcount3 );
  /* verify that previous elements were preserved correctly. */
  succ = rarr_get(lr, 5, &contents);
  assert( succ==0 );
  assert( contents.value == testcount1 );
  succ = rarr_get(lr, 10, &contents);
  assert( succ==0 );
  assert( contents.value == testcount2 );


  /* downsize the array. */
  succ = rarr_downsize( lr );
  assert( lr->capacity == orig_cap );

  rarr_destroy( lr );
  free( lr );

  printf("Success.\n\n");

  return;
}

void run_bloomfilter_tests() {

  printf("=== Running Bloom filter tests. ===\n");

  bloomfilter* bf = malloc( sizeof(bloomfilter) );
  /* make a big bloom filter. */
  int succ = bloomfilter_init(bf, 1024, 32);

  if( succ==-1 || bf==NULL ) {
    printf("Memory allocation failed in Bloom filter test.\n\n");
    return;
  }

  bloomfilter_insert(bf, "hello world", 11);
  assert(bloomfilter_query(bf, "hello waldo", 11) == false);
  assert(bloomfilter_query(bf, "hello world", 11) == true);

  /* test that we can serialize to files correctly. */


  bloomfilter_destroy(bf);
  free(bf);

  printf("Success.\n\n");
  return;
}

int main( ) {
  srandom(1970);

  /**************************************************************
   *								*
   *	 Tests for lowl_math.c 					*
   *								*
   **************************************************************/
  test_bool("powposint 2^0", powposint(2, 0) == 1);
  test_bool("powposint 2^1", powposint(2, 1) == 2);
  test_bool("powposint 2^10", powposint(2, 10) == 1024);
  test_bool("powposint 2^11", powposint(2, 11) == 2048);
  test_bool("powposint 2^31", powposint(2, 31) == 2147483648);

  /* TODO
	Write code to verify that the code to retrieve primes is working.
	Verify that all such numbers are indeed prime (probably best done
		by just checking online).	*/

  /**************************************************************
   *								*
   *	 Tests for lowl_hash.c 					*
   *								*
   **************************************************************/

  // TODO test mod_fnv

  run_multip_add_shift_tests();

  run_motwani_tests();

  /**************************************************************
   *								*
   *	 Tests for lowl_bloom.c 					*
   *								*
   **************************************************************/

  run_bloomfilter_tests();

  /******************************************************
   *							*
   *	Tests for resizable arrays.			*
   *							*
   ******************************************************/

  run_resizablearray_tests();

  printf("All tests completed.\n");
  return 0;
}
