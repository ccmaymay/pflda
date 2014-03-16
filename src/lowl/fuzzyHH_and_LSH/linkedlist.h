#ifndef LINKEDLIST_H
#define LINKEDLIST_H

typedef struct LinkedList{
  int size; /* number of elements in the list. */

  /* head and tail of the list. */
  LLNode* head;
  LLNode* tail;

  /* function for destroying data. */
  void (*destroy)(void* data);

  /* function for comparing elements. */
  //int (*match)(const void* data1, const void* data2);
} LinkedList;


typedef struct LLNode{
  /* data contained in this node. */
  void* data;

  /* pointer to the next list element. NULL if no such element. */
  LLNode* next;
}

/* initialize a new linked list. */
void LinkedList_init(LinkedList* ll, void (*destroy)(void* data));

void LinkedList_destroy(LinkedList* ll);

/* insert given data after the given node. */
int LinkedList_insertNext(LinkedList* ll, LLNode* node, const void* data);

/* remove the element after the one specified by the caller;
put its content into the given pointer.
node=NULL means to remove the head element. */
int LinkedList_removeNext(LinkedList* ll, LLNode* node, void** data);

/* macros for accessing attributes. */
#define LinkedList_size(LLptr) ((LLptr)->size)
#define LinkedList_head(LLptr) ((LLptr)->head) 
#define LinkedList_tail(LLptr) ((LLptr)->tail) 
#define LinkedList_isHead(LLptr, elmt) ((elmt) == ((LLptr)->head) ? 1 : 0 )
#define LinkedList_isTail(elmt) ((elmt)->next == NULL ? 1 : 0 )
#define LLNode_data(elmt) ((elmt)->data)
#define LLNode_next(elmt) ((elmt)->next)

#endif
