#include <stdlib.h>
#include <string.h>
#include "lshnode.h"

void LSHNode_init(LSHNode* lshn, int depth, int breadth) {
  lshn->depth = depth;
  lshn->breadth = breadth;

  /* note that when we first initialize we're temporarily breaking the
	invariant that the following two arrays are NULL under specific
	necessary/sufficient conditions.
	THIS IS AN INCONSISTENCY THAT WE OUGHT TO FIX, BUT THE PROBLEM
	IS THAT WE ONLY KNOW THAT THIS IS A LEAF VS INTERNAL NODE
	BY QUERYING THE TREE THAT IT IS CONTAINED IN.
	Maybe we can avoid needing to rely on that?	*/
  lshn->children = malloc(beadth*sizeof(void*));
  memset( lshn->children, 0, sizeof(void*) );
  lshn->elements = malloc(sizeof(LinkedList));

  lshn->parent = malloc(sizeof(LSHNode*));

  return;
}

void LSHNode_destroy(LSHNode* lshn) {
  /* If we are destroying this node, we must also destroy all of its
	children. This does mean that we cannot have more than one
	tree pointing to the same node, and we should be careful to
	avoid that, though I don't think there is a use case where that
	would actually happen. */
  
  int i;
  if( lshn->children != NULL ) {
    for(i=0; i<LSHNode_breadth(lshn); i++) {
      LSHNode_destroy( (lshn->children)[i] );
      free( (lshn->children)[i] );
    }
  }
  free(lshn->children);

  /* the key is that LinkedList_destroy frees the memory associated with
	the pointers to elements, but not with the elements themselves. */
  LinkedList_destroy(lshn->elements);
  free(lshn->elements); 

  return;
}

/* insert an element at this node. Return success code.
	If this is a terminal node in the tree (i.e. it is at maximum depth)
	or if there is not yet an element stored at this node,
	then add the element to the list of elements stored at this node.
	If this node is not already (i.e., it is not at the maximum depth) */
int LSHNode_insert(LSHNode* lshn, void* elmt, LSHOracle* lsho,
		int maxDepth, int treeID ) {
  int depth = LSHNode_depth(lshn);

  if( LSHNode_isEmpty(lshn) || depth >= maxDepth ) {
    LSHNode_storeElmt(lshn, elmt);
    return 0;

  }  else { /* This is an internal node, which has an element
		already stored at it, and has no children.
		We have to split this node to separate these two elements.  */
    int newElmtHashVal = LSHOracle_evalHash(lsho, treeID, depth, elmt);

    (lshn->children)[newElmtHashVal] = malloc(sizeof(LSHNode));
    if( (lshn->children)[newElmtHashVal] == NULL ) {
      return -1;
    }
    LSHNode_init( (lshn->children)[newElmtHashVal], depth+1,
				LSHNode_breadth(lshn));
    ( (lshn->children)[newElmtHashVal] )->parent = lshn;
    /* now we have to remove the element from this node's pool and hash it to
	the next lowest depth. It is possible that this hashes to the same
	node as the one we just created, in which case we pretty much repeat
	this whole condition again. */
    /* there's some weird pointer manipulation that has to happen */
    void* oldElmt;
    /* only have to remove one element, because if this isn't a terminal
	node then it's only allowed to have one child, anyway. */
    /* LinkedList_removeNext(LinkedList* ll, LLNode* node, void** data)
	NULL argument means remove head of the list.
	After function call, oldElmt points to the data that was held in
	that node. */
    LinkedList_removeNext( lshn->children, NULL, &oldElmt );
    /* it remains to hash this element to the next level down
	and insert it in the correct child node. */
    int oldElmtHashVal = LSHOracle_evalHash(lsho, treeID, depth, oldElmt);
    if( (lshn->children)[oldElmtHashVal] == 0 ) {
      /* don't already have a node there. */
      LSHNode_init( (lshn->children)[oldElmtHashVal], depth+1,
				LSHNode_breadth(lshn) ) ;
      ( (lshn->children)[oldElmtHashVal] )->parent = lshn;
    }
    /* LSHNode_insert(LSHNode* lshn, void* elmt, LSHOracle* lsho,
                int maxDepth, LSHFunctionID** lshfnIDs) */
    return LSHNode_insert( (lshn->children)[oldElmtHashVal], oldElmt,
					lsho, maxDepth, lshfnIDs );
  }
}

/* Run queryElmt down this node. Make *(result) point to the node that
	the query element ends up at. */
void LSHNode_descend(LSHNode* lshn, void* query, LSHNode** result,
			LSHOracle* lsho, LSHFunctionID** lshfnIDs) {
  if( LSHNode_isEmpty(lshn) ) {
    *result = lshn;
  } else {
    /* hash this element and send it down the appropriate child of lshn. */
    int depth = LSHNode_depth(lshn);
    int hashval = LSHOracle_evalHash(lsho, lshfnIDs[depth], query);
    LSHNode_descend( (lshn->children)[hashval], query, result, lsho, lshfnIDs);
  }
  return;
}
