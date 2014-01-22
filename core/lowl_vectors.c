#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
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

/********************************************************
 *							*
 *	Functions that operate on numeric vectors.	*
 *							*
 ********************************************************/

svec_entry svec_entry_from_compvalpair( unsigned int comp, float val ) {
  svec_entry sve;
  sve.component = comp;
  sve.value = val;
  return sve;
}

//int numeric_vector_is_sparse_vector( numeric_vector_ptr numvec ) {
//  /* return 1 if the numeric_vector stored at the given pointer is a
//	sparse vector.
//	Return 0 otherwise.	*/
//
//  /* dense vectors have their entries given by a simple array of floats.
//	sparse vectors have entries given by an array of svec_entry,
//	which are two cells each of size at least sizeof(float), so
//	it suffices to check whether the contents of the entries array
//	is the size of floats or larger.	*/
//  if( sizeof( *((numvec.dense)->entries) ) == sizeof(float) ) {
//    return 0;
//  } else {
//    return 1;
//  }
//}
//
//int numeric_vector_is_dense_vector( numeric_vector_ptr numvec ) {
//  /* return 1 if the numeric vector is dense, 0 otherwise. */
//  
//  if( sizeof( *((numvec.dense)->entries) ) == sizeof(float) ) {
//    return 1;
//  } else {
//    return 0;
//  }
//}
//
//float numeric_vector_get_component(numeric_vector_ptr numvec,
//					unsigned int comp) {
//  /* get the value in the component comp of the given vector. */
//  if( numeric_vector_is_sparse_vector( numvec ) ) {
//    return sparse_vector_get_component( numvec.sparse, comp);
//  } else {
//    return dense_vector_get_component( numvec.dense, comp);
//  }
//}

int dense_vector_init( dense_vector* dv, float* entries, unsigned int len ) {

  dv->entries = malloc( len*sizeof(float) );

  if( dv->entries == NULL ) {
    return LOWLERR_BADMALLOC;
  }
  memcpy( dv->entries, entries, len*sizeof(float) );

  dv->length = len;

  return 0;
}

int sparse_vector_init(sparse_vector* sv, unsigned int* components,
                float* values, unsigned int sparsity, unsigned int len) {

  if( sparsity > len ) {
    return LOWLERR_BADINPUT;
  }

  sv->sparsity = sparsity;
  sv->entries = malloc( sv->sparsity*sizeof(svec_entry) );
  if( sv->entries == NULL ) {
    return LOWLERR_BADMALLOC;
  }
  sv->length = len;

  unsigned int i;
  for( i=0; i<sv->sparsity; i++ ) {
    if( components[i] >= len ) {
      return LOWLERR_BADINPUT;
    }
    sv->entries[i] = svec_entry_from_compvalpair( components[i], values[i] );
  }

  sparse_vector_sort_entries( sv );

  return 0;
}

int sparse_vector_sort_entries( sparse_vector* sv ) {
  /* sort the entries in this sparse vector by increasing component number */

  int succ;
  succ = sparse_vector_sort_entries_helpersort(sv->entries, 0, sv->sparsity-1);

  return succ;
}

int sparse_vector_sort_entries_helpersort( svec_entry* entries,
				unsigned int start, unsigned int end ) {
  /* merge sort the entries in the given array at locations start to end,
	inclusive. Sort by increasing components. */
  if( start==end ) {
    return 0; /* base case.*/
  } else {
    unsigned int mid = (start+end)/2;
    sparse_vector_sort_entries_helpersort( entries, start, mid );
    sparse_vector_sort_entries_helpersort( entries, mid+1, end );
    sparse_vector_sort_entries_helpermerge( entries, start, mid+1, end );
    return 0;
  }
} 

int sparse_vector_sort_entries_helpermerge( svec_entry* entries,
			unsigned int startfirst, unsigned int startsecond,
			unsigned int end ) {
  /* merge the sorted entries from index startfirst to startsecond-1
	with the sorted entries from startsecond to end. */
  svec_entry* scratch_pad = malloc( (end-startfirst+1)*sizeof(svec_entry) );
  if( scratch_pad==NULL ) {
    return LOWLERR_BADMALLOC;
  }
  unsigned int aa = 0;
  unsigned int ii = startfirst;
  unsigned int jj = startsecond;
  while( ii <= startsecond-1 && jj <= end ) {
    if( entries[ii].component <= entries[jj].component ) {
      scratch_pad[aa] = entries[ii];
      ++ii;
    } else {
      scratch_pad[aa] = entries[jj];
      ++jj;
    }
    ++aa;
  }
  if( aa <= end-startfirst+1 ) {
    /* we didn't copy all the elements from the two arrays into the sorted
	array. The fact that we left the while loop means that either
	ii > startsecond-1 or jj > end. Check which of the two conditions
	DOESN'T hold, and copy the remaining elements from the corresponding
	array into the merged array stored in scratch_pad. */
    if( ii <= startsecond-1 ) {
      while( ii <= startsecond-1 ) {
        scratch_pad[aa++] = entries[ii++];
      }
    } else {
      while( jj <= end ) {
        scratch_pad[aa++] = entries[jj++];
      }
    }
  }
  /* copy the sorted list back into place. */
  memcpy( entries+startfirst, scratch_pad,
		(end-startfirst+1)*sizeof(svec_entry) );

  free( scratch_pad );
  return 0;
}

float dense_vector_get_component( dense_vector *dv, unsigned int comp ) {
  /* return the value in component comp. */
  assert( comp < dv->length );
  return (dv->entries)[comp];
}

float sparse_vector_get_component( sparse_vector *sv, unsigned int comp ) {
  /* return the value in component comp. */
  assert( comp < sv->length );

  /* binary search it up!
	In the future, it might be helpful to replace this with a data
	structure that allows constant-time lookup-- is there some variant
	of a heap that will let us keep the property of being able to traverse
	the elements in a nicely sorted order and do constant-time lookup
	without using a hash map?
	It isn't particularly urgent, since the sparsity is assumed small
	relative to the length of the vector, but worth bearing in mind. */

  /* keep track of the range in which the element could lie. */
  unsigned int lowerbound = 0;
  unsigned int upperbound = sv->sparsity - 1;
  unsigned int i = (lowerbound+upperbound)/2;

  while( lowerbound <= upperbound ) {
    if( (sv->entries)[i].component==comp ) {
      return (sv->entries)[i].value;
    } else if( comp > (sv->entries)[i].component ) {
      lowerbound=i+1;
    } else {
      upperbound = i-1;
    }
    i = (lowerbound+upperbound)/2;
  }
  /* comp wasn't among the non-zero components of the vector. */
  return 0.0;
}

//float numeric_vector_dot_product( numeric_vector_ptr nvaa,
//					numeric_vector_ptr nvbb ) {
//  /* return the inner product of the two vectors.
//	Note that this version is a lot less efficient than the methods
//	available to us if we know that both vectors are sparse
//	or that both vectors are dense */
//  if( numeric_vector_is_sparse_vector( nvaa ) ) {
//    if( numeric_vector_is_sparse_vector( nvbb ) ) {
//      return sparse_vector_dot_product( nvaa.sparse, nvbb.sparse );
//    } else { // nvaa is sparse but nvbb is dense.
//      return sparsedense_vector_dot_product( nvaa.sparse, nvbb.dense );
//    }
//  } else { // nvaa is dense.
//    if ( numeric_vector_is_dense_vector( nvbb ) ) {
//      return dense_vector_dot_product( nvaa.dense, nvbb.dense );
//    } else { // nvaa is dense, nvbb is sparse.
//      return sparsedense_vector_dot_product( nvbb.sparse, nvaa.dense );
//    }
//  }
//}

float sparse_vector_dot_product( sparse_vector* svaa, sparse_vector* svbb ) {
  /* return the inner product of the two given sparse vectors. */

  /* it suffices to simply traverse the entry lists of the two vectors,
	since both lists are sorted by component index. */
  unsigned int iaa=0;
  unsigned int ibb=0;
  float dotproduct = 0.0;

  while( iaa < svaa->sparsity && ibb < svaa->sparsity ) {
    if( svaa->entries[iaa].component == svbb->entries[ibb].component ) {
      dotproduct += svaa->entries[iaa].value * svbb->entries[ibb].value;
      ++iaa;
      ++ibb;
    } else if ( svaa->entries[iaa].component > svbb->entries[ibb].component ) {
      ++ibb;
    } else {
      ++iaa;
    }
  }
  return dotproduct;
}  

float dense_vector_dot_product( dense_vector* dvaa, dense_vector* dvbb ) {
  /* compute the dot product of two dense vectors. */

  /* deal "gracefully with dimensionality mismatch by using the dimensionality
	of the "shorter" vector. */
  unsigned int length;
  length = (dvaa->length <= dvbb->length) ? dvaa->length : dvbb->length;

  unsigned int i;
  float dotproduct = 0.0;
  for(i=0; i<length; i++ ) {
    dotproduct += (dvaa->entries)[i]*(dvbb->entries)[i];
  }

  return dotproduct;
}

float sparsedense_vector_dot_product( sparse_vector* spa, dense_vector* den ) {
  /* compute the dot product between two numeric vectors, one or which is
	sparse and the other dense. */

  /* it suffices to just traverse the non-zero entries of the sparse vector.*/
  unsigned int ii, component;
  float dotproduct = 0.0;
  for(ii=0; ii < spa->sparsity; ii++ ) {
    component = spa->entries[ii].component;
    if( component >= den->length ) {
      /* deal "gracefully" with mismatched dimensionality by using the dim.
	of the "shorter" vector. */
      break;
    }
    dotproduct += spa->entries[ii].value * den->entries[component];
  }
  return dotproduct;
}

void sparse_vector_print_components( sparse_vector* spa ) {
  /* print the non-zero components */
  unsigned int i;
  for(i=0; i < spa->sparsity; i++ ) {
    printf("%d : %f\n", spa->entries[i].component, spa->entries[i].value );
  }
  return;
}

void sparse_vector_destroy( sparse_vector* sv ) {
  if( sv->entries != NULL ) {
    free( sv->entries );
  }
  sv->entries = NULL;
  sv->sparsity = 0;
  sv->length = 0;
}

void dense_vector_destroy( dense_vector* dv ) {
  if( dv->entries != NULL ) {
    free( dv->entries );
  }
  dv->length = 0;
}
