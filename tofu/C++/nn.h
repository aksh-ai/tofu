#include "layers.h"

typedef struct graph 
{
  Module layer;
  graph *next;
  graph *prev;
};

graph *Node()
{
  return (graph *)malloc(sizeof(graph));
}

graph *Node(Module layer)
{
  graph *node = (graph *)malloc(sizeof(graph));
  
  node->layer = layer;
  node->next = NULL;
  node->prev = NULL;

  return node;
}