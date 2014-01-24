#include "lowl_math.h"

#include <stdlib.h>
#include <assert.h>

/* We're going to keep a prime around for each power of 2 up to some
	reasonable value. For each power of 2, we'd like a prime that is
	bigger than that power of 2, but not much bigger (so that it still
	fits in a machine word of the same size).
	So lowlmath_usefulprimes[i] = <a prime that is bigger than 2^i.>
	It isn't actually clear whether or not this is useful to have,
	but let's keep it for now.	*/

unsigned int get_useful_prime( unsigned int i ) {
  unsigned int usefulprimes[32] =
		{2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031, 2053,
		4099, 8209, 16411, 32771, 65537, 1310077, 262147, 524309,
		1048583, 2097169, 4194319, 8388617, 16777259, 33554467,
		67108879, 134217757, 268435459, 536870923, 1073741827,
		2147483659 };
  return usefulprimes[i];
}

unsigned long powposint(unsigned long base, unsigned int pow) {
  /* return base raised to power. */
  if( pow==0 ) {
    return 1;
  } else if( pow==1 ) {
    return base;
  } else if( pow % 2 == 1 ) { // if an odd power
    return base*powposint( base*base, (pow-1)/2 );
  } else { // even power.
    return powposint( base*base, pow/2 );
  }
}

long randint(long n) {
  // return uniform value between 0 (inclusive) and n (exclusive)
  // by rejection sampling

  if (n <= 0 || n - 1 > RAND_MAX)
    return -1;

  if (n - 1 == RAND_MAX)
    return random();

  assert(n <= RAND_MAX);

  // Now we're being a little inefficient because we are going to
  // pretend random() has a range of [0, RAND_MAX) instead of
  // [0, RAND_MAX].  This should not bias the samples though.)

  const long bucket_size = RAND_MAX / n;
  const long acceptance_bound = n * bucket_size;
  long r = 0;
  while (1) { // oh yeah
    r = random();
    if (r < acceptance_bound)
      return r / bucket_size;
  }
}

void shuffle(unsigned int *x, size_t n) {
  for (size_t i = n-1; i > 0; --i) {
    long j = randint(i + 1);
    unsigned int temp = x[i];
    x[i] = x[j];
    x[j] = temp;
  }
}
