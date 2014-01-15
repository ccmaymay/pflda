#ifndef LOWLSAMPLE_H
#define LOWLSAMPLE_H

#define min(a,b) ((a) < (b)) ? (a) : (b)

#include "lowl_types.h"
#include <stddef.h>
#include <stdio.h>

typedef struct reservoirsampler {
  size_t capacity;
  size_t stream_pos;
  lowl_key* sample;
} reservoirsampler;

int reservoirsampler_init(reservoirsampler* rs, size_t capacity);
void reservoirsampler_destroy(reservoirsampler* rs);
int reservoirsampler_insert(reservoirsampler* rs, lowl_key x, size_t *idx,
  int *ejected, lowl_key *ejected_key);
size_t reservoirsampler_capacity(reservoirsampler* rs);
size_t reservoirsampler_occupied(reservoirsampler* rs);
void reservoirsampler_print(reservoirsampler* rs);
void reservoirsampler_write(reservoirsampler* rs, FILE* fp);
int reservoirsampler_read(reservoirsampler* rs, FILE* fp);
int reservoirsampler_sample(reservoirsampler* rs, size_t *idx);

#endif
