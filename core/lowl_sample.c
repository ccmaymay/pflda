#include "lowl_sample.h"

/*********************************************************
 *                                                       *
 * Reservoir sampler                                     *
 *                                                       *
 *********************************************************/

int reservoirsampler_init(reservoirsampler* rs, size_t capacity) {
  rs->capacity = capacity;
  rs->stream_pos = 0;
  rs->sample = malloc(sizeof(lowl_key) * capacity);
  if (rs->sample == 0)
    return -1;
  return 0;
}

void reservoirsampler_destroy(reservoirsampler* rs) {
  free(rs->sample);
}

int reservoirsampler_add(reservoirsampler* rs, lowl_key x, size_t *idx,
    lowl_key *ejected) {
  if (rs->occupied < rs->capacity) {
    *idx = rs->occupied;
    ejected = 0;
    rs->sample[*idx] = x;
    ++(rs->occupied);
    return 0;
  } else {
    *idx = random() % rs->stream_pos; // TODO not uniform
    if (*idx < rs->capacity) {
      *ejected = rs->sample[*idx];
      rs->sample[*idx] = x;
      return 0;
    } else {
      *idx = rs->capacity + 1;
      ejected = 0;
      return 1;
    }
  }
}

size_t reservoirsampler_capacity(reservoirsampler* rs) {
  return rs->capacity;
}

size_t reservoirsampler_occupied(reservoirsampler* rs) {
  return min(rs->occupied, rs->capacity);
}
