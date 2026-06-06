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
    free(t->data);
    free(t->shape);
    free(t->strides);
    free(t); 
};



