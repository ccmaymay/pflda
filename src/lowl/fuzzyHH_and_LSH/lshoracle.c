#include "lshoracle.h"

void LSHOracle_init(LSHOracle* lsho, int numTrees, int maxDepth, int hashRange) {
  /* arr of ptrs to LSH fns*/ 
  lsho->functions = LSHFunctionPtr[numTrees][depth];
  lsho->numTrees = numTrees;
  lsho->depth = maxDepth;
  lsho->hashRange = hashRange;
}

/* evaluate the function at depth j in the i-th tree of the forest
        on the given element. */
int LSHOracle_evalHash(*lsho, int treeID, int depth, void* elmt) {
  /* retrieve the correct function. */
  LSHFunctionPtr lshfnptr;
  lshfnptr = ((lsho)->functions)[treeID][depth];
  /* evaluate that hash on this element.
	lshfnptr is a pointer to a function, so we use it as-is
	(no need to get contents) */
  return (lshfnptr)(*elmt);
} 



#endif
