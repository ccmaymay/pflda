/*
Keith Levin
Johns Hopkins University
klevin@jhu.edu
September, 2013

Read vectors of floats from input file (or stdin) and print them,
one per line.

Receives input from a file, if indicated, or reads from stdin (by default).	

Input:
	dim
		Dimensionality of the vectors.

*/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "util.h"
#include "distances.h"

#define DEBUG_MODE_ON 1

char* ipfilename = NULL; /* specifies file to read from (if applicable). */
FILE* ifile; /* pointer to the file we are reading from. */
int dim = 1; /* dimensionality of input vectors/signatures. */

void usage();
void parse_args(int argc, char** argv);
int read_next_vector(float* vector, int d, FILE* fptr);

void usage()
{
  fatal("usage: print_points [-dim <int> (REQUIRED)]\n\
\t[-ifile <string> ]\n");
}

void parse_args(int argc, char **argv) {
  int i;
  for( i = 1; i < argc; i++ )
  {
     if ( strcmp(argv[i], "-ifile") == 0 ) ipfilename = argv[++i];
     else if ( strcmp(argv[i], "-dim") == 0 ) dim = atoi(argv[++i]);
     else {
       fprintf(stderr, "unknown arg: %s\n", argv[i]);
       usage();
     }
  }
  /* check that given parameters are in the allowable ranges. */
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
  /* check more stuff. worry about it later. */
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  /* read vectors from input until we hit the end of the stream */
  float *vec = MALLOC(dim*sizeof(float));


  if( DEBUG_MODE_ON ) {
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
      print_vector(vec, dim);
      printf("\n");
      numvecs++;
    }
  }
  printf("Last value of c was %d.\n", c);
  printf("Read %ld vectors from input.\n", numvecs);

  FREE(vec);
}

int read_next_vector(float* vector, int d, FILE* fptr) {
  //size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream); 
  int s = fread(vector, sizeof(float), d, fptr);
  return s;
}

