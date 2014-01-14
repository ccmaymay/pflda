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
  rs->sample = NULL;
}

bool reservoirsampler_insert(reservoirsampler* rs, lowl_key x, size_t *idx,
    lowl_key *ejected) {
  if (rs->stream_pos < rs->capacity) {
    *idx = rs->stream_pos;
    ejected = 0;
    rs->sample[*idx] = x;
    ++(rs->stream_pos);
    return true;
  } else {
    *idx = random() % rs->stream_pos; // TODO not uniform
    if (*idx < rs->capacity) {
      *ejected = rs->sample[*idx];
      rs->sample[*idx] = x;
      return true;
    } else {
      *idx = rs->capacity + 1;
      ejected = 0;
      return false;
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
  printf("%u", rs->sample[0]);
  size_t occupied = reservoirsampler_occupied(rs);
  for (size_t i = 1; i < occupied; ++i)
    printf(" %u", rs->sample[i]);
  for (size_t i = occupied; i < rs->capacity; ++i)
    printf(" ()");
  printf("\n");
}

void reservoirsampler_write(reservoirsampler* rs, FILE* fp) {
  fwrite( &(f->capacity), sizeof(size_t), 1, fp);
  fwrite( &(f->stream_pos), sizeof(size_t), 1, fp);
  fwrite( f->sample, sizeof( lowl_key ), f->size, fp);
  fwrite( &(f->hash_key_to_word1), sizeof( char_hash ), 1, fp);
  fwrite( &(f->hash_key_to_word2), sizeof( char_hash ), 1, fp);
  fwrite( &(f->hash_key_to_bit1), sizeof( char_hash ), 1, fp); 
  fwrite( &(f->hash_key_to_bit2), sizeof( char_hash ), 1, fp); 
}

void reservoirsampler_read(reservoirsampler* rs, FILE* fp) {
}

void bloomfilter_write(bloomfilter* f, FILE* fp) {
}

void bloomfilter_read(bloomfilter* f, FILE* fp) {
  f->mask = (uint32_t*) malloc(32 * sizeof(uint32_t));
  bloomfilter_setmask( f->mask );

  fread(&(f->size), sizeof(int), 1, fp);
  fread(&(f->k), sizeof(int), 1, fp);
  f->b = (uint32_t*)malloc( f->size*sizeof(uint32_t));
  fread(f->b, sizeof(uint32_t), f->size, fp);
  fread( &(f->hash_key_to_word1), sizeof( char_hash ), 1, fp);
  fread( &(f->hash_key_to_word2), sizeof( char_hash ), 1, fp);
  fread( &(f->hash_key_to_bit1), sizeof( char_hash ), 1, fp);
  fread( &(f->hash_key_to_bit2), sizeof( char_hash ), 1, fp);
}
