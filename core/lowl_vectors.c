#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "lowl_vectors.h"

int bitvector_init(bitvector* bv, unsigned int numbits) {
  bv->numbits = numbits;
  /* allocate memory for the bitvector.
        Each char takes care of sizeof(char) of the bits we need,
        so divide the requested number of bits by the number of bits
        in a single char, check for the case where numbits isn't a multiple
        of 8, and allocate the required amount of memory. */
  unsigned int nchars_to_alloc = numbits/(8*sizeof(char));
  if( numbits % 8*sizeof(char) != 0 ) {
    nchars_to_alloc++;
  }
  bv->bits = malloc( nchars_to_alloc*sizeof(char) );

  if( bv->bits==NULL ) {
    return LOWLERR_BADMALLOC;
  }

  bitvector_clear( bv );

  return LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS;
}

void bitvector_find_indices( unsigned int loc, unsigned int* charindex,
					unsigned int* bitindex ) {
  /* given a location (an index in a bit vector), we often need to
	know which char in the array of chars where we actually store
	the bits corresponds to this location.
	We also need to know which bit of that char corresponds to the
	given location.
	So, figure out that char and bit. Write them to the address
	pointed to by charindex and bitindex, respectively.
	To give a concrete example, consider the following bit vector:
	0001 1001 0100 0010.
	It has length=16.
	Suppose we want to access the bit at location 10 (recall that
	we use 0-indexing, so location 10 is the 11th bit in the array).
	Then this location corresponds to
	charindex=1 (the second bit-- again, 0-indexing)
	and bitindex=2 (again, and I can't stress this enough, 0-indexing).
	location=0 would give bitindex and charindex both 0.	*/
  *charindex = loc/(8*sizeof(char));
  *bitindex = loc % (8*sizeof(char));

  return;
}

int bitvector_set( bitvector* bv, char byte,
                        unsigned int start, unsigned int nbits ) {
  /* set the first nbits starting at start in the given bitvector
	to the first nbits of the given char byte.
	Bits are counted from the right of the byte. So the 3rd bit of
	00010101 is 1.

	Return an error code.	*/

  if( start+nbits-1 >= bv->numbits ) {
    return LOWLERR_BADINPUT;
  }

  unsigned int charindex,bitindex;
  bitvector_find_indices( start, &charindex, &bitindex );

  /* the picture looks like this:
        start: 11, nbits: 7,
        bitvector: 00101101 10111100 00100001, byte: 00111010
                               XXXXX XX
	Bits with X under them are goingto be set. The result will look like
	bitvector: 00101101 10100111 01100001
	NOTE: This picture assumes that we count bits left to right.
	Unfortunately, the actual way that we count bits on the machine
	does not conform to this picture, which is quite unfortunate,
	but that's the way the cookie crumbles. The mental model
	behaves this way, and if you print a bitvector, it will look
	like this picture. */

  /* figure out how many of the bits we are setting lie in one char
	and how many lie in the second one. We have to do this because it's
	possible that the run of bits we're setting actually spans two
	chars in the bitvector's underlying representation. */
  unsigned int nbits_in_first_char, nbits_in_second_char;
  if( bitindex + nbits <= 8*sizeof(char) ) {
    nbits_in_first_char = nbits;
    nbits_in_second_char = 0;
  } else {
    nbits_in_second_char = (nbits + bitindex) % (8*sizeof(char));
    nbits_in_first_char = nbits - nbits_in_second_char;
  }

  /* set the bits in the first char. */
  unsigned char mask = 0;
  unsigned int i;
  for(i=0; i<nbits_in_first_char; i++) {
    mask = (mask << 1) | 1;
  }
  /* now left-shift the whole mask so that we select the upper bits. */
  mask = (mask << bitindex);
  (bv->bits)[charindex] &= ~mask; /*zero the bits that we're about to set.*/
  (bv->bits)[charindex] |= (mask & (byte << bitindex));

  if( nbits_in_second_char==0 ) {
    return 0; /* no bits need to be set inthe second char */
  }
  for(i=0; i<nbits_in_second_char; i++) {
    mask = ((mask << 1) | 1);
  }
  (bv->bits)[charindex+1] &= ~mask; /*zero the bits we're about to set. */
  (bv->bits)[charindex+1] |= (mask & byte);

  return 0;
}

int bitvector_on( bitvector* bv, unsigned int loc) {
  /* set the bit at location loc to 1.
	return a success code.	*/
  if( loc >= bv->numbits ) { // location out of range.
    return LOWLERR_BADINPUT;
  }

  unsigned int charindex, bitindex;
  bitvector_find_indices( loc, &charindex, &bitindex );

  char mask = (1 << bitindex);
  (bv->bits)[charindex] = (bv->bits)[charindex] | mask;
  return LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS;
}

int bitvector_off( bitvector* bv, unsigned int loc) {
  /* set the bit at location loc to 0.
        return a success code.  */
  if( loc >= bv->numbits ) { // location out of range.
    return LOWLERR_BADINPUT;
  }

  unsigned int charindex, bitindex;
  bitvector_find_indices( loc, &charindex, &bitindex );

  /* mask selects the bit we care about, but we want to set that bit to 0
	so we do bitwise-AND with the COMPLEMENT of the bitmask,
	which leaves all other bits in the char intact. */
  char mask = (1 << bitindex);
  (bv->bits)[charindex] = (bv->bits)[charindex] & ~mask;
  return LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS;
}

int bitvector_flip( bitvector* bv, unsigned int loc) {
  /* flip the bit at location loc.
	i.e., if the bit at loc is 0, set it to 1. if it's 1, set to 0.
        return a success code.  */
  if( loc >= bv->numbits ) { // location out of range.
    return LOWLERR_BADINPUT;
  }

  unsigned int charindex, bitindex;
  bitvector_find_indices( loc, &charindex, &bitindex );

  /* mask selects the bit we care about. */
  char mask = (1 << bitindex);
  /* all bits in mask that we DON'T care about are set to 0, so XORing
	them with the other bits in the char that we don't care about
	just leaves that bits as-is.
	The bit we care about in mask is set to 1, so XORing will flip
	the bit in the bitvector.	*/
  (bv->bits)[charindex] = (bv->bits)[charindex] ^ mask;
  return LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS;
}

lowl_bool bitvector_lookup(bitvector* bv, unsigned int loc) {

  if( loc>= bv->numbits ) { // location must be 0<=loc<length.
    return LOWLERR_BADINPUT;
  }

  /* we want to know which char our desired location is in,
        and then which bit of thatchar it's in. */
  unsigned int charindex, bitindex;
  bitvector_find_indices( loc, &charindex, &bitindex );

  char mask = (1 << bitindex);

  if( ( (bv->bits)[charindex] & mask) == 0 ) {
    return FALSE; /* no bit in this position. */
  } else {
    return TRUE; /* found a bit in this position. */
  }
}

void bitvector_clear( bitvector* bv ) {
  /* set all bits in the vector to 0 */
  unsigned int numchars = (bv->numbits)/(8*sizeof(char));
  if( bv->numbits % (8*sizeof(char)) != 0 ) {
    ++numchars;
  }
  memset( bv->bits, 0, numchars*sizeof(char) );
}

void bitvector_print( bitvector* bv ) {
  /* print the bits of the bitvector in human-readable format. */
  unsigned int i;
  for(i=0; i<bv->numbits; ++i) {
    printf( "%d",bitvector_lookup(bv, i) );
  }
  printf("\n");
}

void bitvector_destroy( bitvector* bv ) {
  if (bv->bits != NULL)
    free( bv->bits );
  bv->bits = NULL;
  bv->numbits = 0;
}
