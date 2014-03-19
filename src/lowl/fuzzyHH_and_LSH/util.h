#ifndef UTIL_H
#define UTIL_H

#include <stdlib.h>

#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif

int same_vectors(int *vec1, int n1, int *vec2, int n2);
void make_cumhist(unsigned int *cumhist, int *hist, int n);
void make_cumhist_signed(int *cumhist, int *hist, int n);
void make_cumhist_long(unsigned long *cumhist, int *hist, int n);

void *MALLOC(size_t sz);
void *CALLOC(size_t nmemb, size_t sz);
void FREE(void *ptr);
int get_malloc_count( void );
long closest_prime( long base );
int min(int a, int b);
int max(int a, int b);
void fatal(char *msg);
void threshold_vector(float *vec, int n, double T);

float *readfeats_file(char *fn, int D, int *fA, int *fB, int *N);
double *readfeats_file_d(char *fn, int D, int *fA, int *fB, int *N);
char *mmap_file(char *fn, int *len);

void tic(void);
float toc(void);

int file_line_count( char * file );

void assert_file_exist( char * file );
void assert_file_write( char * file );

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

#endif
