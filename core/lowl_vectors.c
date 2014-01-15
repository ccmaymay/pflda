#include <stdlib.h>
#include <lowl_vectors.h>


int bitvector_init(bitvector* bv, unsigned int numbits) {
  bv->numbits = numbits;
  /* allocate memory for the bitvector.
        Each char takes care of sizeof(char) of the bits we need,
        so divide the requested number of bits by the number of bits
        in a single char, check for the case where numbits isn't a multiple
        of 8, and allocate the required amount of memory. */
  unsigned int nbytes_to_alloc;
  nchars_to_alloc = numbits/(8*sizeof(char));
  if( numbits % 8*sizeof(char) != 0 ) {
    nchars_to_alloc++;
  }
  bv->bits = malloc( nchars_to_alloc*sizeof(char) );

  if( bv->bits==NULL ) {
    return LOWLERR_BADMALLOC;
  }

  return 0;
}

int bitvector_lookup(bitvector* bv, unsigned int loc) {

  if( loc>= bv->numbits ) { // location must be 0<=loc<length.
    return LOWLERR_BADINPUT;
  }

  /* we want to know which char our desired location is in,
        and then which bit of thatchar it's in. */
  unsigned int charindex = loc/(8*sizeof(char));
  unsigned int bitindex = loc % (8*sizeof(char));
  char mask = (1 << bitindex);
  if( ( (bv->bits)[charindex] | mask) == 0 ) {
    return 0; /* no bit in this position. */
  } else {
    return 1; /* found a bit in this position. */
  }
}

void bitvector_clear( bitvector* bv ) {
  /* set all bits to 0 */
  unsigned int numchars = (bv->numbits)/(8*sizeof(char));
  if( numbits % (8*sizeof(char)) != 0 ) {
    ++numchars;
  }
  memset( bv->bits, 0, numchars*sizeof(char) );
}
