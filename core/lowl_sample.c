#include "lowl_sample.h"
#include <stdlib.h>

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
  rs->sample = NULL;
}

int reservoirsampler_insert(reservoirsampler* rs, lowl_key x, size_t *idx,
    lowl_key *ejected) {
  if (rs->stream_pos < rs->capacity) {
    *idx = rs->stream_pos;
    ++(rs->stream_pos);
    ejected = 0;
    rs->sample[*idx] = x;
    return TRUE;
  } else {
    *idx = random() % rs->stream_pos; // TODO not uniform
    ++(rs->stream_pos);
    if (*idx < rs->capacity) {
      *ejected = rs->sample[*idx];
      rs->sample[*idx] = x;
      return TRUE;
    } else {
      *idx = rs->capacity + 1;
      ejected = 0;
      return FALSE;
    }
  }
}

size_t reservoirsampler_capacity(reservoirsampler* rs) {
  return rs->capacity;
}

size_t reservoirsampler_occupied(reservoirsampler* rs) {
  return min(rs->stream_pos, rs->capacity);
}

void reservoirsampler_print(reservoirsampler* rs) {
  printf("%u", (unsigned int) rs->sample[0]);
  size_t occupied = reservoirsampler_occupied(rs);
  for (size_t i = 1; i < occupied; ++i)
    printf(" %u", (unsigned int) rs->sample[i]);
  for (size_t i = occupied; i < rs->capacity; ++i)
    printf(" ()");
  printf("\n");
}

void reservoirsampler_write(reservoirsampler* rs, FILE* fp) {
  fwrite( &(rs->capacity), sizeof(size_t), 1, fp);
  fwrite( &(rs->stream_pos), sizeof(size_t), 1, fp);
  fwrite( rs->sample, sizeof( lowl_key ), reservoirsampler_occupied(rs), fp);
}

int reservoirsampler_read(reservoirsampler* rs, FILE* fp) {
  fread( &(rs->capacity), sizeof(size_t), 1, fp);
  fread( &(rs->stream_pos), sizeof(size_t), 1, fp);
  rs->sample = malloc(sizeof(lowl_key) * rs->capacity);
  if (rs->sample == 0)
    return -1;
  fread( rs->sample, sizeof( lowl_key ), reservoirsampler_occupied(rs), fp);
  return 0;
}
