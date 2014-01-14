#ifndef LOWLSAMPLE_H
#define LOWLSAMPLE_H

#define min(a,b) ((a) < (b)) ? (a) : (b)

#include <stdbool.h>
#include "lowl_types.h"

typedef struct reservoirsampler {
  size_t capacity;
  size_t stream_pos;
  lowl_key* sample;
} reservoirsampler;

int reservoirsampler_init(reservoirsampler* rs, size_t capacity);
void reservoirsampler_destroy(reservoirsampler* rs);
int reservoirsampler_add(reservoirsampler* rs, lowl_key x, size_t *idx,
  lowl_key *ejected);
size_t reservoirsampler_capacity(reservoirsampler* rs);
size_t reservoirsampler_occupied(reservoirsampler* rs);

#endif
