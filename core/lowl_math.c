#include "lowl_math.h"

/* We're going to keep a prime around for each power of 2 up to some
	reasonable value. For each power of 2, we'd like a prime that is
	bigger than that power of 2, but not much bigger (so that it still
	fits in a machine word of the same size).
	So lowlmath_usefulprimes[i] = <a prime that is bigger than 2^i.>
	It isn't actually clear whether or not this is useful to have,
	but let's keep it for now.	*/

unsigned int get_useful_prime( unsigned int i ) {
  unsigned int lowlmath_usefulprimes[32] =
		{2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031, 2053,
		4099, 8209, 16411, 32771, 65537, 1310077, 262147, 524309,
		1048583, 2097169, 4194319, 8388617, 16777259, 33554467,
		67108879, 134217757, 268435459, 536870923, 1073741827,
		2147483659 };
  return lowlmath_usefulprimes[i];
}

unsigned long lowlmath_powposint(unsigned long base, unsigned int pow) {
  /* return base raised to power. */
  if( pow==0 ) {
    return 1;
  } else if( pow==1 ) {
    return base;
  } else if( pow % 2 == 1 ) { // if an odd power
    return base*lowlmath_powposint( base*base, (pow-1)/2 );
  } else { // even power.
    return lowlmath_powposint( base*base, pow/2 );
  }
}
