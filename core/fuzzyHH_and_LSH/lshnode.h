#ifndef LSHNODE_H
#define LSHNODE_H

typedef struct LSHNode {
  int depth; /* depth of this node. Needed to index into array of LSH fns? */
  int breadth; /* max number of children that this node has. */

  /* parent node to this one. NULL if this is a root. */
  LSHNode* parent;

  /* array of children of this node (NULL iff this is a leaf node.) */ 
  LSHNode** children;

  /* List of points stored at this node.
	Empty iff this is an internal node. */ 
  LinkedList* elements;
} LSHNode;

#define LSHNode_depth(lshn) ((lshn)->depth)
#define LSHNode_breadth(lshn) ((lshn)->breadth)
#define LSHNode_parent(lshn) ((lshn)->parent)

/* initialize this LSHNode. */
void LSHNode_init(LSHNode* lshn, int depth, int breadth);

/* destroy LSHNode. */
void LSHNode_destroy(LSHNode* lshn);

/* insert the given element at this node. Return success code. */ 
int LSHNode_insert(LSHNode* lshn, void* elmt, LSHOracle* lsho,
		int maxDepth, LSHFunctionID** lshfnIDs);

/* run query element down from this node. Point result at that node. */
void LSHNode_descend(LSHNode* lshn, void* query, LSHNode** result,
				LSHOracle lsho, LSHFunctionID** lshfnIDs);

/* append all descendants of this node to the given LinkedList.
	Return the number of descendants added. */
int LSHNode_appendDescendants(LSHNode* lshn, LinkedList* ll);

#endif
