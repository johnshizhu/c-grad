#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stdbool.h>
#include "../include/tensor.h"

typedef enum { 
    SUM, 
    MAX, 
    MEAN 
} ReduceOp;


// prototypes
Tensor *dim_reduce(Tensor *t, int dim, ReduceOp op); 
Tensor *elementwise_add(Tensor *t1, Tensor *t2); 
Tensor *hadamard_product(Tensor *t1, Tensor *t2); 
Tensor *matrix_product(Tensor *t1, Tensor *t2); 

#endif
