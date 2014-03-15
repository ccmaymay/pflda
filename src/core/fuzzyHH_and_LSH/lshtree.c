#include <stdlib.h>
#include "lshtree.h"

void LSHTree_init(LSHTree* lsht, int hashRange, int maxDepth) {
  lsht->maxDepth = maxDepth;
  lsht->currentDepth = 0;
  lsht->root = NULL;

  /* allocate memory for storing pointers to our K LSH functions. */
  lsht->lshfnIDs = malloc(K*sizeof(*LSHFunctionID));

  /* Not sure if we need this now that we have done refactoring.
  // generate K LSH function IDs; store them in the array.
  GenerateLSHFunctionIDs(lshoracle, K, lsht->lshfnIDs); */

  /* keep track of the range of set hash functions. */
  lsht->d = LSHOracle_getHashRange(lshoracle);

  return;
} 

void LSHTree_destroy(LSHTree* lsht) {
  int i;
  for(i=0; i<lsht->maxDepth; i++) {
    LSHFunctionID_destroy((lsht->lshfns)[i]);
  }
  free(lsht->lshfnIDs);

  /* we also need to destroy all the nodes in the tree. Note that
	we want to destroy the nodes (and free associated memory), but
	we don't want to blow away the data they point to, since that data
	is pointed to by several different trees. */
  LSHNode_destroy(lsht->root);
  free(lsht->root);

  memset(lsht, 0, sizeof(LSHTree));
  return;
}

/* Run queryElmt down this tree. Point result at the node that we
	end up at and return the depth of that node. */
void LSHTree_descend(LSHTree* lsht, void* queryElmt, LSHNode** result,
			SeedArray* seedarray, int** treeID) {

  return LSHNode_descend(lsht->root, queryElmt, result, seedarray, treeID);
}

/* insert an element into an LSH tree. We must pass the ID of this tree
	along with the hash oracle to allow us to look up the
	hash functions at each level of the tree evaluated on the
	new element. */
int LSHTree_insert(LSHTree* lsht, void* elmt, int treeID, LSHOracle* lsho ) {
  /* initial insert of element into tree hashes at depth=0. */
  int hashOutput = LSHOracle_evalHash(lsho, treeID, 0, elmt);
  int hashRange = LSHOracle_getHashRange(lsho);
  int successCode; /* for returning information, eventually. */

  if( lsht->root == NULL) { /* create a root node, if necessary. */
    lsht->root = malloc( sizeof(LSHNode) );
    if( lsht->root == NULL ) {
      return -1; /* failure to allocate root node memory*/
    }
    LSHNode_init(lsht->root,0,hashRange); /* root node is at depth 0. */
  }
  successCode = LSHNode_insert(lsht->root, elmt,
			lsho, LSHTree_maxDepth(lsht), treeID, lsho);

  return successCode;
} 
