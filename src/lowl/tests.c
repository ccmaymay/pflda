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
#include "lowl_vectors.h"

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
    keys[i] = (unsigned int) randint(m);
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

void run_bitvector_tests() {

  printf("=== Running bitvector tests. ===\n");

  /* check that our lookups of bit indices works correctly. */
  unsigned int bitindex, charindex;

  bitvector_find_indices(0, &charindex, &bitindex);
  assert( bitindex == 0 );
  assert( charindex == 0 );

  bitvector_find_indices(1, &charindex, &bitindex);
  assert( bitindex == 1 );
  assert( charindex == 0 );

  bitvector_find_indices(8, &charindex, &bitindex);
  assert( bitindex == 0 );
  assert( charindex == 1 );

  bitvector_find_indices(7, &charindex, &bitindex);
  assert( bitindex == 7 );
  assert( charindex == 0 );

  bitvector_find_indices(15, &charindex, &bitindex);
  assert( bitindex == 7 );
  assert( charindex == 1 );
  
  bitvector *bv = malloc( sizeof(bitvector) );
  if( bv==NULL ) {
    printf("Memory allocation failed in bitvector test.\n\n");
    return;
  }
  unsigned int nbits = 100;
  int succ = bitvector_init(bv, nbits);
  if( succ==LOWLERR_BADMALLOC ) {
    printf("Memory allocation failed in bitvector test.\n\n");
    return;
  }

  /* verify that all bits are 0. */
  unsigned int i;
  for(i=0; i<nbits; i++) {
    assert( bitvector_lookup(bv, i)==0 );
  }

  /* try settings some bits to 1. */
  bitvector_on( bv, 10 );
  assert( bitvector_lookup(bv, 10)==1 );
  bitvector_on( bv, 99 );
  assert( bitvector_lookup(bv, 99)==1 );
  assert( bitvector_lookup(bv, 11) == 0 );
  assert( bitvector_lookup(bv, 9) == 0 );
  assert( bitvector_lookup(bv, 98) == 0 );
  /* make sure error gets thrown correctly. */
  assert( bitvector_lookup(bv, 100) == LOWLERR_BADINPUT );

  /* try setting some bits to 0. */
  bitvector_off( bv, 1 ); // this was already off.
  bitvector_off( bv, 10 ); // this was previously on.
  assert( bitvector_lookup(bv, 1) == 0 );
  assert( bitvector_lookup(bv, 10) == 0 );

  /* flip some bits. */
  bitvector_flip( bv, 1);
  bitvector_flip( bv, 99); 
  assert( bitvector_lookup(bv, 1)==1 );
  assert( bitvector_lookup(bv, 99)==0 );

  /* set multiple bits at once. */
  bitvector_set( bv, 15, 55, 3 );
  printf("\n");
  assert( bitvector_lookup(bv, 55)==1 );
  assert( bitvector_lookup(bv, 56)==1 );
  assert( bitvector_lookup(bv, 57)==1 );
  assert( bitvector_lookup(bv, 58)==0 );

  /* verify that this still works when the bits we're setting cross
	the boundaries of two chars.  */
  bitvector_set( bv, 127, 70, 6);
  assert( bitvector_lookup( bv, 69)==0 );
  assert( bitvector_lookup( bv, 70)==1 );
  assert( bitvector_lookup( bv, 71)==1 );
  assert( bitvector_lookup( bv, 72)==1 );
  assert( bitvector_lookup( bv, 73)==1 );
  assert( bitvector_lookup( bv, 74)==1 );
  assert( bitvector_lookup( bv, 75)==1 );
  assert( bitvector_lookup( bv, 76)==0 );

  bitvector_destroy( bv );
  free(bv); 

  printf("Success.\n\n");
  return;
}

void run_numeric_vector_tests() {
  printf("=== Running numeric vector tests. ===\n");

  sparse_vector* spavecaa = malloc( sizeof(sparse_vector) );
  sparse_vector* spavecbb = malloc( sizeof(sparse_vector) );
  dense_vector* denvecaa = malloc( sizeof(dense_vector) );
  dense_vector* denvecbb = malloc( sizeof(dense_vector) );

  float vals[6] = { 2.2f, 3.3f, 5.5f, 8.8f, 11.11f, 19.19f };
  unsigned int comps[6] = {2, 3, 5, 8, 11, 19};
  dense_vector_init( denvecaa, vals, 6); 
  assert( denvecaa->length == 6 );
  sparse_vector_init( spavecaa, comps, vals, 6, 20);
  assert( spavecaa->length == 20);
  assert( spavecaa->sparsity == 6);

  /* test that our mergesort for sparse vector entries works properly. */
  unsigned int comps2[7] = { 6, 4, 3, 1, 5, 2, 0 };
  float vals2[7] = { 60.6f, 40.4f, 30.3f, 10.1f, 50.5f, 20.2f, 0.0f };
  sparse_vector_init( spavecbb, comps2, vals2, 7, 20 );
  assert( spavecbb->length == 20 );
  assert( spavecbb->sparsity == 7 );

  /* verify that entries are sorted */
  unsigned int i;
  for( i = 0; i<6; i++ ) {
    assert( (spavecbb->entries)[i].component
		< (spavecbb->entries)[i+1].component );
    if( i==5 ) {
      continue;
    }
    assert(((spavecaa->entries)[i]).component
		< ((spavecaa->entries)[i+1]).component );
  } 

  /* initialize a second dense vector. */
  dense_vector_init( denvecbb, vals2, 6); 
  assert( denvecbb->length == 6 );

  /* test dot products */
  float dotprod = sparse_vector_dot_product( spavecaa, spavecbb );
  float expected_dotprod = 20.2f*2.2f + 30.3f*3.3f + 50.5f*5.5f;
  assert( fabsf(dotprod - expected_dotprod ) < 0.0001f );

  dotprod = dense_vector_dot_product( denvecaa, denvecbb );
  expected_dotprod = 2.2f*60.6f + 3.3f*40.4f + 5.5f*30.3f + 8.8f*10.1f
				+ 11.11f*50.5f + 19.19f*20.2f;
  assert( fabsf(dotprod - expected_dotprod) < 0.0001f );

  dotprod = sparsedense_vector_dot_product( spavecaa, denvecaa );
  expected_dotprod = 5.5f*2.2f + 8.8f*3.3f + 19.19f*5.5f;
  assert( fabs( dotprod - expected_dotprod ) < 0.0001f );

  /* check that component retrieval is correct. */
  assert( fabsf(dense_vector_get_component( denvecaa, 0 ) - 2.2f) < 0.0001f );
  assert( fabsf(sparse_vector_get_component( spavecaa, 3) - 3.3f) < 0.0001f );
  assert( fabsf(sparse_vector_get_component( spavecaa, 0) - 0.0f) < 0.0001f );
  assert( fabsf(sparse_vector_get_component( spavecaa, 2) - 2.2f) < 0.0001f );
  assert( fabsf(sparse_vector_get_component( spavecaa, 8) - 8.8f) < 0.0001f );
  assert( fabsf(sparse_vector_get_component( spavecaa, 9) - 0.0f) < 0.0001f );

  /* deallocate memory. */
  sparse_vector_destroy( spavecaa );
  sparse_vector_destroy( spavecbb );
  dense_vector_destroy( denvecaa );
  dense_vector_destroy( denvecbb );

  free( spavecaa );
  free( spavecbb );
  free( denvecaa );
  free( denvecbb );

  printf("Success.\n\n");
  return;
}

void run_bloomfilter_tests() {
  printf("=== Running Bloom filter tests. ===\n");

  bloomfilter* bf = malloc( sizeof(bloomfilter) );
  /* make a big bloom filter. */
  int succ = bloomfilter_init(bf, 1024, 32);

  if( succ!=0 || bf==NULL ) {
    printf("Init failed in Bloom filter test.\n\n");
    return;
  }

  bloomfilter_insert(bf, "hello world", 11);
  assert(bloomfilter_query(bf, "hello waldo", 11) == FALSE);
  assert(bloomfilter_query(bf, "hello world", 11) == TRUE);

  /* test that we can serialize to files correctly? */

  bloomfilter_destroy(bf);
  free(bf);

  printf("Success.\n\n");
  return;
}

void run_ht_key_to_count_tests() {
  printf("===Running ht_key_to_count_tests. ===\n");

  ht_key_to_count* ht = malloc( sizeof(ht_key_to_count) );

  if( ht==NULL ) {
    printf("Memory allocation failed in ht_key_to_count test.\n\n");
    return;
  }

  int succ =  ht_key_to_count_init( ht, 256 );
  if( succ==LOWLERR_BADMALLOC ) {
    printf("Memory allocation failed in initializing ht_key_to_count.\n");
    return;
  }
  assert( succ==0 );

  /* try adding an element to the table. */
  lowl_key current_key = 1901;
  lowl_count current_val = 666;
  succ = ht_key_to_count_set( ht, current_key, current_val );
  assert( succ==LOWLHASH_NOTINTABLE );

  ht_key_to_count_destroy( ht );
  free( ht );

  printf("Success.\n\n");
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

  /**************************************************************
   *								*
   *	 Tests for lowl_hash.c 					*
   *								*
   **************************************************************/

  run_multip_add_shift_tests();

  run_motwani_tests();

  run_ht_key_to_count_tests();

  /**************************************************************
   *								*
   *	 Tests for lowl_bloom.c 				*
   *								*
   **************************************************************/

  run_bloomfilter_tests();

  /**************************************************************
   *								*
   *	Tests for lowl_vectors.c				*
   *								*
   **************************************************************/
  run_bitvector_tests();
  run_numeric_vector_tests();

  printf("All tests completed.\n");
  return LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS;
}