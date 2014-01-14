#ifndef DISTANCES_H
#define DISTANCES_H

/* functions for computing distances between vectors and bit strings. */

// float (*distfn)(void* vector1, void* vector2, int d);

/* euclidean vector-vector distance */
float euc_vec_dist(void* vector1, void* vector2, int d);

/* (exact) cosine distance between vectors. */
float cos_vec_dist(void* vector1, void* vector2, int d);

/* Hamming distance between signatures. */
float ham_sig_dist(void* sig1, void* sig2, int siglen);

/* Approximate cosine distance between signatures. */
float apxcos_sig_dist(void* sig1, void* sig2, int siglen);

/* print contents of vector. */
void print_vector(float* vector, int d);

#endif
