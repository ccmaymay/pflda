/*
Keith Levin
Johns Hopkins University
klevin@jhu.edu
September, 2013

Implements a simple reservoir sampling-based baseline solution
to the fuzzy heavy hitters problem, in which we see a stream of vectors
and wish to return a set of vectors, which includes, for each cluster
representing at least proportion \alpha of the stream, at least one
vector from that cluster, with possible additional vectors,
none of which come from clusters of proportion less than some
yet-to-be-determined proportion \alpha^\prime.

Parameters:
	alpha - desired threshold proportion.
	epsilon - size of the ball around cluster centers.

Receives input from a file, if indicated, or reads from stdin (by default).	

Additional options:
	Sampling mode
		Select between "hopping" and "coin flip" reservoir sampling.
	Distance measure
		Select between cosine and Euclidean distance.
	LSH signature input
		Treat inputs as LSH bit signatures (and thus compute distance
		as cosine).
*/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "util.h"
#include "reservoir.h"
#include "distances.h"

#define DEBUG_MODE_ON 1

float alpha; /* alpha parameter; HH proportion. */
float epsilon; /* epsilon parameter; ball radius. */
char* ipfilename = NULL; /* specifies file to read from (if applicable). */
FILE* ifile; /* pointer to the file we are reading from. */
char* samplemode = NULL; /* specifies sampling strategy to use. */
char* distfn_name = NULL; /* specifies distance function. */
/* pointer to the function that we will use for distane between vectors,
signatures, whatever. */
float (*distfn)(void* vector1, void* vector2, int d);
int use_sigs = 0; /* if set to 1, input is LSH signatures. */
int dim = 1; /* dimensionality of input vectors/signatures. */

void usage();
void parse_args(int argc, char** argv);
int read_next_vector(float* vector, int d, FILE* fptr);
void output_heavy_hitters(Reservoir* res);
int handle_new_element(Reservoir* res, void* elmt, long* counts, long* drops);
int choose_capacity(float alpha, float epsilon);

void usage()
{
  fatal("usage: fuzzyhh [-alpha <float> (REQUIRED)]\n\
\t[-epsilon <float> (REQUIRED)]\n\
\t[-ifile <string> ]\n\
\t[-samplemode <string> (defaults to \'hop\')]\n\
\t[-distfn <string> (defaults to euclidean)]\n\
\t[-dim <int> (defaults to 1) ]\n");
}

void parse_args(int argc, char **argv) {
  int i;
  for( i = 1; i < argc; i++ )
  {
     if( strcmp(argv[i], "-alpha") == 0 ) alpha = atof(argv[++i]);
     else if ( strcmp(argv[i], "-epsilon") == 0 ) epsilon = atof(argv[++i]);
     else if ( strcmp(argv[i], "-ifile") == 0 ) ipfilename = argv[++i];
     else if ( strcmp(argv[i], "-samplemode") == 0 ) samplemode = argv[++i];
     else if ( strcmp(argv[i], "-distfn") == 0 ) distfn_name = argv[++i];
     else if ( strcmp(argv[i], "-dim") == 0 ) dim = atoi(argv[++i]);
     else {
       fprintf(stderr, "unknown arg: %s\n", argv[i]);
       usage();
     }
  }
  /* check that given parameters are in the allowable ranges. */
  if( alpha <= 0.0 || alpha > 1.0 ) {
    printf("alpha must be in the range (0,1].\n");
    usage();
  }
  if( epsilon <= 0.0 ) {
    printf("epsilon must be a positive number.\n");
    usage();
  }
  if( dim <= 0 ) {
    printf("dimensionality must be nonnegative.\n");
    usage();
  }
  if( ipfilename != NULL ) {
    assert_file_exist( ipfilename );
    ifile = fopen(ipfilename, "r");
  } else {
    ifile = stdin;
  }
  if( distfn_name == NULL ) {
    printf("Using Euclidean distance.\n");
    distfn = &euc_vec_dist;
  }
  if( samplemode == NULL ) {
    samplemode = "hop";
  }
  /* check more stuff. worry about it later. */
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  /* decide on a capacity for the reservoir. */
  int cap = choose_capacity(alpha,epsilon);

  /* Initialize a reservoir to sample into. */
  Reservoir res;
  /* may need to change elementSize, or write functionality to choose
  elmtSize on the fly, especially if we want to make this handle both
  float and double inputs. */
  int elmtSize = dim*sizeof(float);
  reservoir_init(&res, cap, elmtSize);

  /* we would also like to keep track of how many points have been matched
  against each element currently in the reservoir. */
  long* slot_counts = MALLOC(cap*sizeof(long));
  memset(slot_counts, 0, cap*sizeof(long));
  long drop_counter = 0; /* count how many samples we dropped. */
 
  /* read vectors from input until we hit the end of the stream */
  float *vec = MALLOC(elmtSize);

  if( DEBUG_MODE_ON ) {
    assert(cap >= 1);
    assert(vec != NULL);
  }

  int c;
  long numvecs = 0; //track how many items we've seen in the stream.
  while( 1 ) {
    c=read_next_vector(vec, dim, ifile);
    /* Check that we haven't reached the end of the file; warn the user if
	we hitthe end of the file without reading the correct number of
	bytes, given the dimensionality information that the user gave us. */
    if ( c == 0 ) {
      break; // no more elements to read from input.
    } else if ( c != dim ) {
      printf("Failing, with c = %d, having read %ld elements.\n", c, numvecs);
      fatal("Reached end of file without reading a whole vector. Are you sure you provided the correct dimensionality?\n");
    } else { // successfully read an element from the stream.
      /* if our policy decides (possibly randomly) to take this sample,
	we do so. */
      handle_new_element(&res, vec, slot_counts, &drop_counter);
      //print_vector(vec, dim);
      numvecs++;
    }
  }
  printf("Last value of c was %d.\n", c);
  printf("Read %ld vectors from input.\n", numvecs);

  output_heavy_hitters(&res);
  printf("\n");
  printf("Slot counts:\n");
  int i;
  for(i=0; i<reservoir_size(&res); i++) {
    printf("%ld points in slot %d\n", slot_counts[i], i);
  }
  printf("Dropped %ld elements.\n", drop_counter);

  FREE(vec);
  FREE(slot_counts);
}

int handle_new_element(Reservoir* resptr, void* elmt, long* counts, long* drop_counter) {
  /* Count this element or add it to a center, or do something else.
	Return an integer code to communicate which action was taken.
  Side effect: update the counts to reflect how many points are now associated
  with each slot of the reservoir.

  0 -- Did nothing, i.e., didn't make the vector a new center nor count
	it as being within a range of another.
  1 -- Filled an empty slot.
  2 -- Element evicted another existing cluster center.
  3 -- Element was within epsilon of an existing cluster center. */

  /* check if this vector falls within epsilon of an existing center. */
  int i;
  for( i=0; i<reservoir_size(resptr); i++ ) {
    /* elsewhere we've made distfn point to the function we want. */
    /* printf("Current element: ");
    print_vector(elmt, dim);
    printf("Comparing against vector in reservoir: ");
    print_vector((float *) &((resptr)->samples[i]), dim);
    printf("Distance to element %d in reservoir: %f\n", i, distfn(elmt, &((resptr)->samples[i]), dim) ); */
    if( distfn(elmt, &(resptr->samples)[i], dim) <= epsilon ) {
      //count_element(element_counts, i);
      //print_vector((float*) elmt, dim);
      //printf("%ld ", counts[i]);
      (counts[i])++;
      //printf("%ld\n", counts[i]);
      return 3; // element was within epsilon, and thus was counted.
    }
  }
  /* If there is an open slot in the reservoir, fill it. */
  if( !reservoir_is_full(resptr) ) {
    int i = reservoir_add_sample(resptr, elmt);
    printf("%d\n",i);
    if( i<0 || i>=reservoir_size(resptr) ) {
      fatal("Something went wrong in adding a sample.\n");
    }
    counts[i] = 1;
    //printf("Added an element, I hope.\n");
    return 1; // filled an empty slot.
  } else {
    /* Perhaps randomly evict an existing element, based on its counts?
	We'll want to look at existing heavy hitter algorithms for
	integer streams to see what they do. */
    (*drop_counter)++;
    
    return 0;
  }
}

int read_next_vector(float* vector, int d, FILE* fptr) {
  //size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream); 
  int s = fread(vector, sizeof(float), d, fptr);
  return s;
}

void output_heavy_hitters(Reservoir * res) {
  printf("===Reservoir summary===\n");
  printf("%d samples in reservoir, out of %d slots.\n", reservoir_size(res), reservoir_capacity(res));
  printf("Contents:\n");
  int i;
  for(i=0; i<reservoir_size(res); i++) {
    print_vector((float*) &(res->samples[i]), dim);
    printf("\n");
  }
}

int choose_capacity(float alpha, float epsilon) {
  int cap = (int) ceilf(1.0/alpha);
  return cap;
}
