

#include "../include/tensor.h"
#include <stdlib.h>
#include <stdbool.h>


bool check_shape_match(Tensor *t1, Tensor *t2) {
    if (t1->ndim != t2->ndim) return false;
    if (t1->size != t2->size) return false; 
    for (int dim = 0; dim < t1->ndim; dim++) {
        if (t1->shape[dim] != t2->shape[dim]) return false; 
    }
    return true; 
}

Tensor *elementwise_add(Tensor *t1, Tensor *t2) {
    if (!check_shape_match(t1, t2)) return NULL;  
    Tensor *res = tensor_create_constant(0, t1->shape, t1->ndim);

    for (int i = 0; i < t1->size; i++) {
        res->data[i] = t1->data[i] + t2->data[i]; 
    }

    return res; 
};

Tensor *hadamard_product(Tensor *t1, Tensor *t2) {
    if (!check_shape_match(t1, t2)) return NULL;  
    Tensor *res = tensor_create_constant(0, t1->shape, t1->ndim);

    for (int i = 0; i < t1->size; i++) {
        res->data[i] = t1->data[i] * t2->data[i];
    }

    return res;
};

bool check_mul_compatible(Tensor *t1, Tensor *t2) { // does not check for broadcasting basic matrix product conditions check 
    int t1_ndim = t1->ndim;
    int t2_ndim = t2->ndim; 
    if (t1_ndim < 2 || t2_ndim < 2) return false;
    int t1_column_count = t1->shape[t1_ndim-1];
    int t2_row_count = t2->shape[t2_ndim-2];

    if (t1_ndim != t2_ndim) return false;  // num dims match 
    if (t1_column_count != t2_row_count) return false; // t1 column match t2 rows
    
    for (int i = 0; i < t1_ndim-2; i++) {    // check that other dims match 
        int t1_dim = t1->shape[i]; 
        int t2_dim = t2->shape[i];
        if (t1_dim != t2_dim) return false; 
    }
    
    return true; 
};

Tensor *matrix_product(Tensor *t1, Tensor *t2) {
    if (!check_mul_compatible(t1, t2)) return NULL; 

    //TODO

    return NULL;
};





