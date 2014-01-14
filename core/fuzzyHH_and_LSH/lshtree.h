#ifndef LSHTREE_H
#define LSHTREE_H

typedef struct LSHTree {
  /* Array of maxDepth hash function IDs, each of which identifies a unique
	hash function from a given family. We pass this ID along with an input
	to be evaluated to an appropriate LSHOracle, which returns the value
	of the ID'd hash function when called on this function.
	This out put is an integer between 0 and d-1.
	This array contains one hash function for each non-terminal
	layer of the tree.*/ 
  LSHFunctionID** lshfnIDs;

  int d; /* size of the range of the hash functions. */

  int maxDepth; /* maximum allowable depth of the tree. */

  /* current depth of the tree. That is, the depth of the deepest leaf
	currently stored in the tree. */ 
  int currentDepth; 

  LSHNode* root; /* root of the tree (depth 0). */
} LSHTree;

#define LSHTree_maxDepth(lsht) ((lsht)->maxDepth)
#define LSHTree_currentDepth(lsht) ((lsht)->currentDepth)
#define LSHTree_hashFunctionRange(lsht) ((lsht)->d)

/* initialize an LSH tree, using functions from the given LSH family.
Tree has maximum depth K. 
Note: you know d from lsh generator (provided you give it that
functionality, which you ought to).*/
void LSHTree_init(LSHTree* lsht, LSHOracle* lshOr, int K);

/* destroy the LSHTree. */
void LSHTree_destroy(LSHTree* lsht);

/* need functions for descending the tree. */
/* Descend the LSHTree *lsht, finding the nearest element to *queryElmt.
	Make *result point to the node where that element resides.   */
void LSHTree_descend(LSHTree* lsht, void* queryElmt, LSHNode** result,
			LSHOracle* lsho);

/* insert a given element into the tree. Return success/failure code. */
int LSHTree_insert(LSHTree* lsht, void* elmt, LSHOracle* lsho );

/*delete a given element from the tree. Return success/failure code. */
int LSHTree_delete( LSHTree* lsht, void* elmt);
