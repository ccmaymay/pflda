#include "lowl_math.h"

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
