#ifndef RESERVOIR_H
#define RESERVOIR_H

typedef struct Reservoir {
  int elementSize; /* size in bytes of a single element in the reservoir. */
  int capacity; /* number of samples that can go in the reservoir */
  int size; /* number of samples taken so far. */
  void** samples; /* array of pointers to our samples. */
} Reservoir;

/* initialize a reservoir of given capacity and element size,
with destructor fn destroy. */
void reservoir_init(Reservoir* res,int capacity,int elmtSize);

/* free the memory associated with the given reservoir. */
void reservoir_destroy(Reservoir* res);

/* figure out how many samples have been taken so far. */
#define reservoir_size(res) ((res)->size)

/* how many samples can this reservoir contain? */
#define reservoir_capacity(res) ((res)->capacity)

/* can we take any more samples? */
#define reservoir_is_full(res) (((res)->size) >= ((res)->capacity) ? 1 : 0 )

/* add this element to the next open space in the reservoir.
Return 0 if successful, 1 otherwise. */
int reservoir_add_sample(Reservoir* res, void* sample);

#endif
