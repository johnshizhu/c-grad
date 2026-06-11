

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


void build_index(int *index, int flat_index, int *shape, int ndim) {
    for (int dim = ndim-1; dim >= 0; dim--) {
        index[dim] = flat_index % shape[dim];
        flat_index = flat_index / shape[dim];
    }
}


typedef enum { REDUCE_SUM, REDUCE_MAX, REDUCE_MEAN } ReduceOp;

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
        float accum = (op == REDUCE_MAX) ? tensor_get(t, in_index) : 0.0f;

        for (int k = 0; k < t->shape[dim]; k++) { // loop over injected dim
            in_index[dim] = k; 
            float val = tensor_get(t, in_index); 

            switch (op) {
                case REDUCE_SUM: accum += val; break; 
                case REDUCE_MAX: accum = val > accum ? val : accum; break; 
                case REDUCE_MEAN: accum += val; break; 
            }
        }

        if (op == REDUCE_MEAN) {
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


Tensor *matrix_product(Tensor *t1, Tensor *t2) {
    if (!check_mul_compatible(t1, t2)) return NULL; 

    int res_ndim = t1->ndim;
    int *res_shape = malloc(res_ndim * sizeof(int));
    res_shape[res_ndim-1] = t2->shape[res_ndim-1];
    res_shape[res_ndim-2] = t1->shape[res_ndim-2];
    for (int i = 0; i < res_ndim-2; i++) {
        res_shape[i] = t1->shape[i];
    }
    Tensor *res = tensor_create_constant(0, res_shape, res_ndim);

    int batch_count = 1; // calculating total batch counts
    for (int i = 0; i < res_ndim-2; i++) {
        batch_count = batch_count * res_shape[i];
    }

    // calculate slice_sizes for t1, t2, and res tensors. 
    int t1_slice_size = t1->shape[t1->ndim-1] * t1->shape[t1->ndim-2];
    int t2_slice_size = t2->shape[t2->ndim-1] * t2->shape[t2->ndim-2];
    int res_slice_size = res->shape[res->ndim-1] * res->shape[res->ndim-2];

    int inner_dim = t1->shape[t1->ndim-1]; // inner dimension

    int t1_index[t1->ndim];
    int t2_index[t2->ndim];
    int res_index[res->ndim];

    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) { // looping over batches 
        
        
        //2d matrix multiplication
        for (int t1_row = 0; t1_row < t1->shape[t1->ndim-2]; t1_row++) { // loop over t1 rows
            for (int t2_column = 0; t2_column < t2->shape[t2->ndim-1]; t2_column++) { // loop over t2 columns
                float sum = 0.0;
                for (int j = 0; j < inner_dim; j++) { // looping over number of inner dim values (same for t1 and t2)
                    int t1_data_index = (batch_idx * t1_slice_size) + (t1_row * t1->shape[t1->ndim-1]) + j;
                    int t2_data_index = (batch_idx * t2_slice_size) + (j * t2->shape[t2->ndim-1]) + t2_column;  
                    sum = sum + (t1->data[t1_data_index] * t2->data[t2_data_index]);
                }
                int res_data_index = (batch_idx * res_slice_size) + (t1_row * res->shape[res_ndim-1]) + t2_column; 
                res->data[res_data_index] = sum;
            }
        }
    }

    return res;
};



