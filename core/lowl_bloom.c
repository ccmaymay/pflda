/* Bloom Filter implementation          */
/* Author: Ashwin Lall                  */
#include <math.h>
#include <stdint.h>
#include "lowl_bloomfilter.h"

void lowl_bloomfilter_init(lowl_bloomfilter* f, int size, int k)
{ 
  int i;

  // These masks help us to access the i-th bit in a uint32_t counter (for i=1...32)
  f->mask = (uint32_t*)malloc(32 * sizeof(uint32_t));
  for(i = 0; i < 8; ++i)
  { 
    f->mask[4 * i + 0] = (1 << 4 * i + 0);
    f->mask[4 * i + 1] = (1 << 4 * i + 1);
    f->mask[4 * i + 2] = (1 << 4 * i + 2);
    f->mask[4 * i + 3] = (1 << 4 * i + 3);
  }

  // we only need size/32 uint32_t since each contributes 32 bits
  f->size = size / 32 + 1;

  f->k = k;
  f->b = (uint32_t*)malloc(f->size * sizeof(uint32_t));

  memset(f->b, 0, f->size * sizeof(uint32_t));

  initHashFunction(&(f->h), f->size);
}

//void lowl_bloomfilter_insert(lowl_bloomfilter* f, char* x, int len)
//{ 
//  int i, index;
//  char y[len + 8];  // assumes that f->k < 10^8
//
//  for(i = 0; i < f->k; ++i)
//  { 
//    sprintf(y, "%d%s", i, x);
//    index = hash(&(f->h), y, strlen(y));
//
//    f->b[index/32] = f->b[index/32] | f->mask[index % 32];
//  }
//}

int lowl_bloomfilter_query(lowl_bloomfilter* f, char* x, int len)
{ 
  int i, index;
  char y[len + 8];  // assumes that f->k < 10^8

  for(i = 0; i < f->k; ++i)
    { 
      sprintf(y, "%d%s", i, x);
      index = hash(&(f->h), y, strlen(y));
      if ((f->b[index/32] & f->mask[index % 32]) == 0)
        return 0;
    }

  return 1;
}

void lowl_bloomfilter_print(lowl_bloomfilter* f)
{
  int i, j;
  for(i = 0; i < f->size; ++i)
    for(j = 0; j < 32; ++j)
      if ((f->b[i] & f->mask[j]) == 0)
        printf("0");
      else
        printf("1");
  printf("\n");
}

void lowl_bloomfilter_write(lowl_bloomfilter* f, FILE* fp)
{
  fwrite(&(f->size), sizeof(int), 1, fp);
  fwrite(&(f->k), sizeof(int), 1, fp);
  fwrite(f->b, sizeof(uint32_t), f->size, fp);
  writeHashFunction(&(f->h), fp);
}

void lowl_bloomfilter_read(lowl_bloomfilter* f, FILE* fp)
{

  int i;

  f->mask = (uint32_t*) malloc(32 * sizeof(uint32_t));
  for(i = 0; i < 8; ++i)
    {
      f->mask[4 * i + 0] = (1 << 4 * i + 0);
      f->mask[4 * i + 1] = (1 << 4 * i + 1);
      f->mask[4 * i + 2] = (1 << 4 * i + 2);
      f->mask[4 * i + 3] = (1 << 4 * i + 3);
    }

  fread(&(f->size), sizeof(int), 1, fp);
  fread(&(f->k), sizeof(int), 1, fp);
  f->b = (uint32_t*)malloc(f->size * sizeof(uint32_t));
  fread(f->b, sizeof(uint32_t), f->size, fp);
  readHashFunction(&(f->h), fp);
}

void lowl_bloomfilter_destroy(lowl_bloomfilter* f)
{
  free(f->b);
  free(f->mask);
}
