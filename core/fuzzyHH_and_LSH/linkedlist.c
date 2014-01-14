#include <string.h>
#include <stdlib.h>
#include "linkedlist.h"

void LinkedList_init(LinkedList* ll, void (*destroy)(void* data)) {
  ll->size = 0;
  ll->head = NULL;
  ll->tail = NULL;
  ll->match = NULL;
  ll->destroy = destroy;

  return;
}

void LinkedList_destroy(LinkedList* ll) {
  void* data;
  if( ll->destroy != NULL) {
    while (LinkedList_size(ll) > 0) {
      if( LinkedList_removeNext(ll, NULL, (void**)&data)==0 ) {
        ll->destroy(data);
      }
    }
  }
  /* clear the memory. */
  memset(ll, 0, sizeof(LinkedList));

  return;
}

int LinkedList_insertNext(LinkedList* ll, LLNode* node, const void* data) {
  LLNode* newNode;/* new node to hold this data. */
  if( (newNode = (LLNode*)malloc(sizeof(LLNode))) == NULL ) {
    return -1; /* failure to allocate memory. */
  }
  newNode->data = (void *)data;

  /* handle cases that can arise when inserting into list. */
  if(node==NULL) { /* insert at head of list. */
    if( LinkedList_size(ll)==0 ) {
      ll->tail = newNode; /* if list is empty, this node is also the tail. */
    }
    newNode->next = ll->head;
    ll->head = newNode;
  } else { /* handle insertion elsewhere in the list. */
    if (node->next == NULL) {
      ll->tail = newNode;
    }
    newNode->next = node->next;
    node->next = newNode;
  }
  /* update list size, return status code. */
  list->size++;

  return 0;
}

int LinkedList_removeNext(LinkedList* ll, LLNode* node, void** data) {
  /* remove the element after the given node. Put its cargo in the given
	pointer **data. */
  LLNode *oldNode;

  if (LinkedList_size(ll) == 0) {
    return -1; /* can't remove from empty list. */
  }

  /* handle valid cases:
	-removal from head of the list
	-removal from elsewhere in the list */
  if (node == NULL) { /* remove from head of list. */
    *data = ll->head->data;
    oldNode = ll->head;
    ll->head = ll->head->next;
    if( LinkedList_size(ll) == 1 ) {
      ll->tail = NULL;
    }
  } else { /* remove from elsewhere. */
    if( node->next == NULL) { /* can't remove from after the tail.*/
      return -1;
    }
    *data = node->next->data;
    oldNode = node->next;
    node->next = (node->next)->next;

    /* if we just removed the tail, we need to establish the new tail. */
    if( node->next == NULL ) {
      ll->tail = node;
    }
  }
  free(oldNode);

  /* resize list and return. */
  list->size--;
  return 0;
}
