#include <string.h>
#include "util.h"
#include "reservoir.h"

/* initialize a reservoir of given capacity, with memory allocated to
hold cap elements of size elementSize. */
void reservoir_init(Reservoir *res, int cap, int elementSize ) {
  /* allocate memory for the samples in the reservoir. */
  /* elementSize is the number of bytes consumed by a single element in the
  reservoir. E.g., if we're storing 10-dimensional vectors of floats,
  elementSize = 10*sizeof(float) */
  res->elementSize = elementSize;
  res->samples = MALLOC(cap*elementSize);
  res->capacity = cap;
  res->size = 0; /* no samples have been taken yet. */
}

/* free the memory associated with the given reservoir. */
void reservoir_destroy(Reservoir *resptr) {

  FREE(resptr->samples);
  memset(resptr, 0, sizeof(Reservoir));

  return;
}

/* add this element to the next open space in the reservoir.
Return the index of the slot that the sample was place in if successful,
-1 otherwise (i.e., if the reservoir was actually full). */
int reservoir_add_sample(Reservoir *res, void* sample) {
  if (reservoir_is_full(res)) return -1;
  int i = reservoir_size(res);
  memcpy(&(res->samples[i]), sample, res->elementSize);;
  (res->size)++;
  return i;
}
