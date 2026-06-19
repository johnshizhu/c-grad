#ifndef OP_H
#define OP_H


typedef struct Tensor Tensor; 

// Node in computation graphfor autograd


typedef enum { 
    ADD,
    HADAMARD_PRODUCT, 
    REDUCE_ADD, 
    RESHAPE, 
    BROADCAST, 
} OpType;


typedef union { 
    struct { int dim; } reduce; 
    struct { int *new_shape; int new_ndim; } reshape; 
    struct { int dim; int val; } broadcast; 
} OpAux; 


typedef struct Op {
    OpType type;
    struct Op *parents[2];
    int num_parents;
    Tensor *out;
    OpAux aux; 
} Op;
// in the case that num_parents = 0, set the out field as raw tensor values


#endif




