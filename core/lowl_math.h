#ifndef LOWLMATH_H
#define LOWLMATH_H

/* bigprime will fit in a 32-bit unsigned int. */
#define LOWLMATH_BIGPRIME 4294967291

/* mathematical tables and functions required by lowl. */

unsigned int get_useful_prime( unsigned int i );

/* raise an integer to a non-negative integer power. */
unsigned long powposint(unsigned long base, unsigned int pow);

long randint(long n);

#endif
