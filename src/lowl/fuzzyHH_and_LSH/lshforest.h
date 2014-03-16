#ifndef LSHFOREST_H
#define LSHFOREST_H

typedef struct LSHForest {
  int forestSize; /* number of trees in the forest. */
  int maxTreeDepth; /* maximum treedepth. */
  LSHTree** trees; /* array of pointers to the L trees in the forest. */

  /* seedArray is a forestSize by maxTreeDepth array of seeds,
	which we use to pick out hash functions. */
  SeedArray* seedArray;
} LSHForest;

/* a SeedArray is an m by n array of pointers to void */
typedef (void*)[][] SeedArray;

#define LSHForest_size(lshf) ((lshf)->forestSize)
#define LSHForest_maxTreeDepth(lshf) ((lshf)->maxTreeDepth)

/* initialize an LSHForest of L trees, each with maximum depth K */
/* somewhere we need to specify a set of hash functions, probably best
to do it here, but perhaps not, in which case we'll have to change this. */
void LSHForest_init(LSHForest* lshf, int L, int K);

/* destroy the LSHForest. */
void LSHForest_destroy(LSHForest* lshf);

/* insert given element into the LSH forest. This means descending each
of the trees, extending paths, if necessary/allowable.
Return an integer code corresponding to what action ended up occurring
(e.g., whether we hit the max depth, whether there was a duplicate node,
etc.  */
int LSHForest_insert(LSHForest* lshf, void* elmt);

/* delete the given element from the LSH forest. This means descending each
of the trees and removing the given element from each of them,
and compressing paths if needed.
Return an integer code capturing what was done (e.g., if the element didn't
exist, etc.) */
int LSHForest_delete(LSHForest* lshf, void* elmt);

/* query for a given element from the LSH forest. Return the m nearest
neighbors to the query element by writing them into the given array.
Return an integer code capturing what happened (e.g., if it was necessary
to return duplicates, or something like that).

Refer to the Bawa et al 2005 paper for suggestions on how to deal with
m > 1 case-- things like dealing with duplicates and stuff. */
int LSHForest_query(LSHForest* lshf, void** elements, int m);

/* synchronous ascend algorithm described in Bawa et al fig. 3 */
/* we have an array of depths and an array of elements. These two
arrays are of length forestSize.
depths[i] is the depth that we found *elements[i] in trees[i].
We simultaneously ascend the L trees in the forest, accumulating a
set of M candidate points, which we put into the given linked list P.
Return an integer code to capture information (e.g., which of the
termination conditions caused us to terminate, etc). */
int LSHForest_synchAscend(LSHForest* lshf, int* depths, void** elements,
				LinkedList* P);

#endif
