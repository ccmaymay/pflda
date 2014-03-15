/* Functions for calculating distances between vectors and signatures. */

#include <math.h>
#include <stdio.h>

/* euclidean vector-vector distance */
float euc_vec_dist(void* vector1, void* vector2, int d) {
  float* v1 = (float*) vector1;
  float* v2 = (float*) vector2;
  float dist = 0.0;
  int i;
  for(i=0; i<d; i++) {
    dist += powf((v1[i] - v2[i]), 2.0);
  }
  return sqrtf(dist);
}

/* (exact) cosine distance between vectors. */
float cos_vec_dist(void* voidaa, void* voidbb, int d) {
  float* aa = (float*) voidaa;
  float* bb = (float*) voidbb;
  float dotprod = 0.0;
  float aanorm = 0.0;
  float bbnorm = 0.0;
  /* cosine(theta) = \frac{<a,b>}{\|a\|*\|b\|} */
  int i;
  for(i=0; i<d; i++) {
    dotprod += aa[i]*bb[i];
    aanorm += aa[i]*aa[i];
    bbnorm += bb[i]*bb[i];
  }
  aanorm = sqrt(aanorm);
  bbnorm = sqrt(bbnorm);
  return dotprod/(aanorm*bbnorm);
}

/* Hamming distance between signatures. */
float ham_sig_dist(void* sig1, void* sig2, int siglen) {
  return 0.0;
}

/* Approximate cosine distance between signatures. */
float apxcos_sig_dist(void* sig1, void* sig2, int siglen) {
  return 0.0;
}

/* print contents of vector. */
void print_vector(float* vector, int d) {
  int i;
  for(i=0; i<d; i++) {
    printf("%f",vector[i]);
    printf((i==d-1) ? "" : " ");
  }
}
