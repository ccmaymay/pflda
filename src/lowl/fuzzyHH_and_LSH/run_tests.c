#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "reservoir.h"
#include "distances.h"

int main(int argc, char**argv) {
  Reservoir res;
  int TEST_RES_CAP = 2;
  int d = 4;

  /* initializing a reservoir with 10 slots should result in a reservoir
  of capacity 10 with size 0 (because no samples have been added yet. */
  reservoir_init(&res, TEST_RES_CAP, d);
  assert(reservoir_capacity(&res) == TEST_RES_CAP);
  assert(reservoir_size(&res) == 0);
  assert(reservoir_is_full(&res) == 0);

  /* adding a single element should increase the size by one. */
  double foo = 1.2345;
  reservoir_add_sample(&res, &foo);
  assert( reservoir_capacity(&res) == TEST_RES_CAP );
  assert( reservoir_size(&res) == 1 );
  assert( !reservoir_is_full(&res) );

  /* add another element. */
  double markofthebeast = 6.66;
  reservoir_add_sample(&res, &markofthebeast);
  assert( reservoir_capacity(&res) == TEST_RES_CAP );
  assert( reservoir_size(&res) == 2 );
  assert( reservoir_is_full(&res) );
  /* add another element. This shouldn't work, because res is full. */
  double third = 0.333333;
  assert( reservoir_add_sample(&res, &third)==-1 );
  assert( reservoir_capacity(&res) == TEST_RES_CAP );
  assert( reservoir_size(&res) == 2 );
  assert( reservoir_is_full(&res) );

  /* verify that destroy() works. */
  reservoir_destroy( &res );

  /* check that our different distance functions operate as intended. */
  float* vec1 = malloc(d*sizeof(float));
  float* vec2 = malloc(d*sizeof(float));
  assert(vec1 != NULL);
  assert(vec2 != NULL);
  int i;
  for(i=0; i<d; i++) {
    vec1[i] = 1.0+i;
    vec2[i] = powf(2.0, i);
  }
  //print_vector(vec1, d);
  //print_vector(vec2, d);

  float truedist = 0.0 + 0.0 + 1.0 + 16.0;
  truedist = sqrtf(truedist);
  assert( fabsf(euc_vec_dist(vec1, vec2, d) - truedist) <= 1e-6 );

  printf("All tests passed.\n");
  return 0;
}
