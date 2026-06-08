#include "../include/tensor.h"
#include <stdlib.h>


int* fresh_strides(int *shape, int ndim) {
    int *strides = malloc(ndim * sizeof(int));
    int cur = 1;
    for (int i = ndim-1; i >= 0; i--) {
        strides[i] = cur; 
        cur = cur * shape[i]; 
    }
    return strides;
}


Tensor* tensor_create_constant(int value, int *shape, int ndim) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size = shape[i] * size;
    }

    Tensor *tensor_ptr = malloc(sizeof(Tensor)); 
    tensor_ptr->dtype = DTYPE_FP32; // hard coded dtype for now
    tensor_ptr->ndim = ndim;
    tensor_ptr->size = size; 
    tensor_ptr->is_view = false; 

    int *shape_cpy = malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape_cpy[i] = shape[i];
    }
    tensor_ptr->shape = shape_cpy;

    float *data_ptr = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        data_ptr[i] = value; 
    }
    tensor_ptr->data = data_ptr;

    tensor_ptr->strides = fresh_strides(tensor_ptr->shape, tensor_ptr->ndim);

    return tensor_ptr;
};


Tensor* tensor_create_rand(int *shape, int ndim) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size = shape[i] * size;
    }

    Tensor *tensor_ptr = malloc(sizeof(Tensor)); 
    tensor_ptr->dtype = DTYPE_FP32; // hard coded dtype for now
    tensor_ptr->ndim = ndim;
    tensor_ptr->size = size; 
    tensor_ptr->is_view = false; 

    int *shape_cpy = malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape_cpy[i] = shape[i];
    }
    tensor_ptr->shape = shape_cpy;

    float *data_ptr = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        data_ptr[i] = (float)rand() / (float)RAND_MAX; 
    }
    tensor_ptr->data = data_ptr;

    tensor_ptr->strides = fresh_strides(tensor_ptr->shape, tensor_ptr->ndim);

    return tensor_ptr;  
};


void tensor_free(Tensor *t) {
    if (!t->is_view) {
        free(t->data);
    }
    free(t->shape);
    free(t->strides);
    free(t); 
};


Tensor *tensor_view(Tensor *t) { 
    Tensor *new_t = malloc(sizeof(Tensor)); 
    new_t->dtype = DTYPE_FP32; 
    new_t->ndim = t->ndim; 
    new_t->size = t->size; 
    new_t->is_view = true; 

    int *new_shape = malloc(new_t->ndim * sizeof(int)); 
    for (int i = 0; i < new_t->ndim; i++) {
        new_shape[i] = t->shape[i]; 
    }
    new_t->shape = new_shape; 

    new_t->data = t->data; 

    int *new_strides = malloc(new_t->ndim * sizeof(int)); 
    for (int i = 0; i < new_t->ndim; i++) {
        new_strides[i] = t->strides[i]; 
    }
    new_t->strides = new_strides; 

    return new_t; 
};


Tensor *tensor_copy(Tensor *t) { // deep copy 
    Tensor *new_t = malloc(sizeof(Tensor)); 
    new_t->dtype = DTYPE_FP32; 
    new_t->ndim = t->ndim; 
    new_t->size = t->size; 
    new_t->is_view = false; 

    int *new_shape = malloc(new_t->ndim * sizeof(int)); 
    for (int i = 0; i < new_t->ndim; i++) {
        new_shape[i] = t->shape[i]; 
    }
    new_t->shape = new_shape; 

    float *new_data = malloc(new_t->size * sizeof(float)); 
    for (int i = 0; i < new_t->size; i++) {
        new_data[i] = t->data[i]; 
    }
    new_t->data = new_data; 

    int *new_strides = malloc(new_t->ndim * sizeof(int)); 
    for (int i = 0; i < new_t->ndim; i++) {
        new_strides[i] = t->strides[i]; 
    }
    new_t->strides = new_strides; 

    return new_t; 
};


bool check_continguous(Tensor *t) { 
    bool contiguous = true; 
    int *exp_strides = fresh_strides(t->shape, t->ndim); 
    for (int i = 0; i < t->ndim; i++) {
        if (t->strides[i] != exp_strides[i]) { // non-continguous 
            contiguous = false; 
        }
    }
    free(exp_strides);
    return contiguous; 
};


Tensor *tensor_continguous(Tensor *t) { // returns a continguous copy of a non-continguous tensor
    bool contiguous = check_continguous(t); 
    if (contiguous) {
        return t; 
    }

    Tensor *new_t = malloc(sizeof(Tensor)); 
    new_t->dtype = DTYPE_FP32; 
    new_t->ndim = t->ndim; 
    new_t->size = t->size; 
    new_t->is_view = false; 

    int *new_shape = malloc(new_t->ndim * sizeof(int)); 
    for (int i = 0; i < new_t->ndim; i++) {
        new_shape[i] = t->shape[i]; 
    }
    new_t->shape = new_shape; 

    float *new_data = malloc(new_t->size * sizeof(float)); 
    int index[t->ndim]; // holds the multi-dim index of values from t
    for (int i = 0; i < new_t->size; i++) {
        int cur_index = i; 
        for (int dim = t->ndim-1; dim >=0; dim--) {
            index[dim] = cur_index % t->shape[dim];
            cur_index = cur_index / t->shape[dim]; 
        }
        new_data[i] = tensor_get(t, index); 
    }
    new_t->data = new_data; 

    new_t->strides = fresh_strides(new_t->shape, new_t->ndim);

    return new_t; 
};


Tensor *tensor_broadcast(Tensor *t, int dim, int broadcast_val) { // set shape to broadcast_val and stride at dim to 0
    if (t->shape[dim] != 1) {
        return NULL; 
    }
    Tensor *new_t = tensor_view(t);
    new_t->shape[dim] = broadcast_val; 
    new_t->strides[dim] = 0; 
    new_t->size = new_t->size * broadcast_val;

    return new_t; 
};


Tensor *tensor_reshape(Tensor *t, int *new_shape, int new_ndim) {
    int new_size = 1; 
    for (int i = 0; i < new_ndim; i++) {
        new_size = new_size * new_shape[i]; 
    }
    if (new_size != t->size) {
        return NULL; 
    }

    bool contiguous = check_continguous(t); 

    // TODO

    if (contiguous) {

    }

    else {

    }

};


Tensor *tensor_transpose() {

};


