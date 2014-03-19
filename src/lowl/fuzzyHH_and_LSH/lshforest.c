#include <stdlib.h>
#include "lshforest.h"
#include "lshtree.h"
#include "lshnode.h"
#include "linkedlist.h"

void LSHForest_init(LSHForest* lshf, int L, int K) {
  lshf->forestSize = L;
  /* it might be easier to just have this be an array of trees, rather
	than an array of pointers to trees, if they memory has to be
	allocated somewhere, anyway... Probably best to keep it as it is,
	since it's more in keeping with style, but worth considering. */
  lshf->trees = malloc(L*sizeof(LSHTree*));
  lshf->maxTreeDepth = K;
  /* for each tree in the forest, allocate an array of pointers to void */
  lshf->seedArray = malloc(L*sizeof(void**));
  /* each such array has K pointers to void in it. */
  int i;
  for(i=0;i<L;i++) {
    (lshf->seedArray)[i] = malloc(K*sizeof(void*));
  }

  return;
}

void LSHForest_destroy(LSHForest* lshf) {
  /* destroy the trees. */
  int i;
  for(i=0; i<lshf->forestSize; i++) {
    LSHTree_destroy((lshf->trees)[i]);
  }
  /* free all the trees. */
  free(lshf->trees);
  /* destroy the seed array. */
  int i;
  for(i=0;i<L;i++) {
    free( (lshf->seedArray)[i] );
  }
  free( lshf->seedArray );
  
  /* clear memory. */
  memset(lshf, 0, sizeof(LSHForest));

  return;
}

int LSHForest_insert(LSHForest* lshf, void* elmt) {
  int i;
  /* insert into each of the L trees. */
  for(i=0; i<lshf->forestSize; i++) {
    LSHTree_insert( (lshf->trees)[i] , elmt, (lshf->seedArray) );
  }

  return 0;
}

int LSHForest_delete(LSHForest* lshf, void* elmt) {
  int i;
  /* delete from each of the L trees. */
  for(i=0; i<lshf->forestSize; i++) {
    LSHTree_delete( (lshf->trees)[i], elmt);
  }

  return 0;
}

void LSHForest_query(LSHForest* lshf, void* q,
	void** elements, int* depths, int m) {
  /* descend each of the trees in the forest, placing, for tree i,
	a pointer to the element returned by the query on tree i
	into (elements)[i] and the corresponding depth into depths[i]. */

  int i;
  /* query each of the L trees. */
  for(i=0; i<lshf->forestSize; i++) {
    LSHTree_query( (lshf->trees)[i], ...);
    depths[i] = LSHTree_descend(LSHTree* lsht, q, &(elements[i]),
					lshf->oracle);
  }

  return 0;
}

int LSHForest_synchAscend(LSHForest* lshf, int* depths, LSHNode** nodes,
                                LinkedList* P, int m) {
  /* simultaneously work our way back up all L trees in the forest.
	depths and elements jointly characterize a set of L locations in
	the forest, one leaf in each tree. nodes[i] is initially a pointer to
	a leaf node, and depths[i] is the depth of that node.
	We might be able to improve things slightly by just using the
	fact that LSHNodes include their depth in their struct.

	We work out way up the forest, accumulating points in linked list
	P until we reach a stopping criterion (e.g., P becomes suitably long).
	The point is that we add points to P in order of depth in
	the forest.   */

  LinkedList_clear(P);
  /* invariant: we'll store the depth we're currently working at in this
	variable. We start at the maximum depth over all trees. */
  int currentDepth = 0;
  int i;
  for(i=0; i<LSHNode_size(lshf); i++) {
    if(depths[i] > currentDepth) {
       currentDepth = depths[i];
    }
  }
  LSHNode** previousNodes = malloc(LSHForest_size(lshf)*sizeof(LSHNode*));
  if( previousNodes == NULL ) {
    return -1;
  }
  while( currentDepth > 0 && LinkedList_size(P) <= m ) {
    for(i=0; i<LSHForest_size(lshf); i++) {
      if( depths[i] == currentDepth ) {
        /* don't descend a node that we've already visited. */
        for(j=0; j<LSHNode_breadth(nodes[i]) ; j++) {
          if( ((nodes[i])->children)[j] != previousNodes[i] ) {
            LSHNode_appendDescendants( ((nodes[i])->children)[j], P );
          }
        }
        /* now update previous to point to the current node, and go up a
		level to the parent node. */
        previousNodes[i] = nodes[i];
        nodes[i] = LSHNode_parent(nodes[i]);
        depths[i]--;
      }
    }
    if( LinkedList_size(P) >= m ) {
      return 0;
    }
    currentDepth--;
  }
  return 0;
}
