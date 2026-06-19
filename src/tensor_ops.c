

#include "../include/tensor.h"
#include "../include/tensor_ops.h"
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>


bool check_shape_match(Tensor *t1, Tensor *t2) {
    if (t1->ndim != t2->ndim) return false;
    if (t1->size != t2->size) return false; 
    for (int dim = 0; dim < t1->ndim; dim++) {
        if (t1->shape[dim] != t2->shape[dim]) return false; 
    }
    return true; 
}


Tensor *dim_reduce(Tensor *t, int dim, ReduceOp op) {
    int *new_shape = malloc((t->ndim - 1) * sizeof(int)); 
    int new_index = 0; 
    for (int i = 0; i < t->ndim; i++) {
        if (i == dim) {
            continue; 
        }
        new_shape[new_index] = t->shape[i]; 
        new_index++; 
    }

    // output tensor has same shape aside from the dim
    Tensor *res = tensor_create_constant(0, new_shape, t->ndim-1);
    free(new_shape); 

    int res_index[res->ndim]; 
    int in_index[t->ndim]; 
    for (int i = 0; i < res->size; i++) { // for each reduction 
        build_index(res_index, i, res->shape, res->ndim); // result index 

        int out_pos = 0; // inject dim back in
        for (int in_dim = 0; in_dim < t->ndim; in_dim++) {
            if (in_dim == dim) {
                continue; 
            }
            in_index[in_dim] = res_index[out_pos];
            out_pos++; 
        }
        
        in_index[dim] = 0;
        float accum = (op == MAX) ? tensor_get(t, in_index) : 0.0f;

        for (int k = 0; k < t->shape[dim]; k++) { // loop over injected dim
            in_index[dim] = k; 
            float val = tensor_get(t, in_index); 

            switch (op) {
                case SUM: accum += val; break; 
                case MAX: accum = val > accum ? val : accum; break; 
                case MEAN: accum += val; break; 
            }
        }

        if (op == MEAN) {
            accum /= t->shape[dim]; 
        }

        tensor_set(res, res_index, accum); 
    }

    return res; 
}


Tensor *elementwise_add(Tensor *t1, Tensor *t2) {
    if (!check_shape_match(t1, t2)) return NULL;  
    Tensor *res = tensor_create_constant(0, t1->shape, t1->ndim);

    int index[t1->ndim];
    for (int i = 0; i < t1->size; i++) {
        build_index(index, i, t1->shape, t1->ndim);
        tensor_set(res, index, (tensor_get(t1, index) + tensor_get(t2, index)));
    }
    
    return res; 
};


Tensor *hadamard_product(Tensor *t1, Tensor *t2) {
    if (!check_shape_match(t1, t2)) return NULL;  
    Tensor *res = tensor_create_constant(0, t1->shape, t1->ndim);

    int index[t1->ndim];
    for (int i = 0; i < t1->size; i++) {
        build_index(index, i, t1->shape, t1->ndim);
        tensor_set(res, index, (tensor_get(t1, index) * tensor_get(t2, index)));
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


Tensor *matrix_product(Tensor *t1, Tensor *t2) { // tinygrad style
    if (!check_mul_compatible(t1, t2)) return NULL; 

    // unsqueeze at leading and trailing ends
    int *t1_new_shape = malloc((t1->ndim + 1) * sizeof(int));
    int *t2_new_shape = malloc((t2->ndim + 1) * sizeof(int));
    memcpy(t1_new_shape, t1->shape, t1->ndim * sizeof(int)); 
    memcpy(t2_new_shape + 1, t2->shape, t2->ndim * sizeof(int)); 
    t1_new_shape[t1->ndim] = 1; 
    t2_new_shape[0] = 1; 
    Tensor *t1_r = tensor_reshape(t1, t1_new_shape, t1->ndim + 1);
    Tensor *t2_r = tensor_reshape(t2, t2_new_shape, t2->ndim + 1); 
    free(t1_new_shape); 
    free(t2_new_shape); 

    // broad cast at leading and training ends
    Tensor *t1_b = tensor_broadcast(t1_r, t1_r->ndim-1, t2_r->shape[t2_r->ndim-1]);
    Tensor *t2_b = tensor_broadcast(t2_r, 0, t1_r->shape[0]);
    tensor_free(t1_r); 
    tensor_free(t2_r); 

    // element-wise multiply 
    Tensor *prod = hadamard_product(t1_b, t2_b); 
    tensor_free(t1_b); 
    tensor_free(t2_b); 

    // summation reduce
    Tensor *res = dim_reduce(prod, prod->ndim - 2, SUM);
    tensor_free(prod); 

    return res; 
};

