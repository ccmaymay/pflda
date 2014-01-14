#ifndef LSHORACLE_H
#define LSHORACLE_H

/* LSH functions map elements (which can have any type, hence the *void)
	to their hashes, which are ints between 0 and d inclusive, where
	d is the breadth of an LSHTree. */
typedef int (*LSHFunctionPtr)(void *);

/* an LSHOracle is an array of such LSH functions and aux. information. */
typedef struct LSHOracle {
  /* array of pointers to functions. */
  LSHFunctionPtr functions[][];
  /* this array is really an l by d matrix-- d hash functions for each
	tree in the forest. */
  int numTrees;
  int depth;
  /* store the range of the hash function, because it's convenient to have. */
  int hashRange;
} LSHOracle;

/* initializea new LSHOracle. */
void LSHOracle_init(LSHOracle* lsho, int size);

/* evaluate the function at depth j in the i-th tree of the forest
	on the given element. */
int LSHOracle_evalHash(*lsho, int treeID, int depth, void* elmt);

/* A hash function returns integers in 0,1,...,d-1. Return d. */ 
#define LSHOracle_getHashRange(lsho) ((lsho)->hashRange)
/* retrieve depth or number of trees. */
#define LSHOracle_getDepth(lsho) ((lsho)->depth)
#define LSHOracle_getNumTrees(lsho) ((lsho)->numTrees)


#endif
