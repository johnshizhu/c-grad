#include "../include/tensor.h"
#include <stdlib.h>
#include <string.h>


void build_index(int *index, int flat_index, int *shape, int ndim) {
    for (int dim = ndim-1; dim >= 0; dim--) {
        index[dim] = flat_index % shape[dim];
        flat_index = flat_index / shape[dim];
    }
}


int* fresh_strides(int *shape, int ndim) {
    int *strides = malloc(ndim * sizeof(int));
    int cur = 1;
    for (int i = ndim-1; i >= 0; i--) {
        strides[i] = cur; 
        cur = cur * shape[i]; 
    }
    return strides;
}


bool check_contiguous(Tensor *t) { 
    bool contiguous = true; 
    int *exp_strides = fresh_strides(t->shape, t->ndim); 
    for (int i = 0; i < t->ndim; i++) {
        if (t->strides[i] != exp_strides[i]) { // non-contiguous 
            contiguous = false; 
            break; 
        }
    }
    free(exp_strides);
    return contiguous; 
};


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
    tensor_ptr->grad = NULL; 
    tensor_ptr->requires_grad = false; 

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
    tensor_ptr->grad = NULL; 
    tensor_ptr->requires_grad = false; 

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
    if (t->grad != NULL) {
        tensor_free(t->grad); 
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
    new_t->requires_grad = t->requires_grad; 
    new_t->grad = NULL; 

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
    new_t->requires_grad = t->requires_grad; 

    int *new_shape = malloc(new_t->ndim * sizeof(int)); 
    memcpy(new_shape, t->shape, t->ndim * sizeof(int)); 
    new_t->shape = new_shape; 

    float *new_data = malloc(new_t->size * sizeof(float)); 

    if (check_contiguous(t)) {
        int *new_strides = malloc(new_t->ndim * sizeof(int)); 
        memcpy(new_strides, t->strides, t->ndim * sizeof(int)); 
        memcpy(new_data, t->data, t->size * sizeof(float)); 
        new_t->strides = new_strides; 
        new_t->data = new_data; 
    }
    else {
        int index[t->ndim];
        for (int i = 0; i < t->size; i++) {
            build_index(index, i, t->shape, t->ndim); 
            new_data[i] = tensor_get(t, index); 
        }
        new_t->strides = fresh_strides(new_t->shape, new_t->ndim); 
        new_t->data = new_data;
    }

    if (t->grad != NULL) { // deep copy of other grad
        Tensor *new_grad = malloc(sizeof(Tensor)); 
        new_grad->dtype = DTYPE_FP32; 
        new_grad->ndim = t->grad->ndim; 
        new_grad->size = t->grad->size; 
        new_grad->is_view = t->grad->is_view; 
        new_grad->requires_grad = t->grad->requires_grad; 
        new_grad->grad = NULL; 

        int *new_grad_shape = malloc(new_t->ndim * sizeof(int)); 
        memcpy(new_grad_shape, new_t->shape, new_t->ndim * sizeof(int)); 
        new_grad->shape = new_grad_shape; 

        int *new_grad_strides = malloc(new_t->ndim * sizeof(int)); 
        memcpy(new_grad_strides, new_t->strides, new_t->ndim * sizeof(int)); 
        new_grad->strides = new_grad_strides; 

        float *new_grad_data = malloc(new_t->size * sizeof(float)); 
        memcpy(new_grad_data, t->grad->data, t->grad->size * sizeof(float)); 
        new_grad->data = new_grad_data; 

        new_t->grad = new_grad; 
    }
    else {
        new_t->grad = NULL; 
    }

    return new_t; 
};


Tensor *tensor_contiguous(Tensor *t) { // returns a contiguous copy of a non-contiguous tensor
    bool contiguous = check_contiguous(t); 
    if (contiguous) {
        return tensor_copy(t); 
    }

    Tensor *new_t = malloc(sizeof(Tensor)); 
    new_t->dtype = DTYPE_FP32; 
    new_t->ndim = t->ndim; 
    new_t->size = t->size; 
    new_t->is_view = false; 
    new_t->requires_grad = t->requires_grad; 
    new_t->grad = NULL; 

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
        new_size *= new_shape[i];
    }
    if (new_size != t->size) {
        return NULL;
    }

    Tensor *new_t = check_contiguous(t) ? tensor_view(t) : tensor_contiguous(t);

    free(new_t->shape);
    new_t->shape = malloc(new_ndim * sizeof(int));
    memcpy(new_t->shape, new_shape, new_ndim * sizeof(int));

    new_t->ndim = new_ndim;

    free(new_t->strides);
    new_t->strides = fresh_strides(new_t->shape, new_t->ndim);

    return new_t;
}


Tensor *tensor_axis_transpose(Tensor *t) { // simple 2d axis transpose, doesn't fit multi-dim
    if (t->ndim != 2) {
        return NULL; 
    }
    Tensor *new_t = tensor_view(t); 
    int last_shape = new_t->shape[new_t->ndim-1];
    int last_stride = new_t->strides[new_t->ndim-1];

    new_t->shape[new_t->ndim-1] = new_t->shape[new_t->ndim-2];
    new_t->strides[new_t->ndim-1] = new_t->strides[new_t->ndim-2];

    new_t->shape[new_t->ndim-2] = last_shape; 
    new_t->strides[new_t->ndim-2] = last_stride;  

    return new_t; 
};


Tensor *tensor_permute(Tensor *t, int *perm, int perm_ndim) {
    if (t->ndim != perm_ndim) {
        return NULL; 
    }
    bool seen[perm_ndim]; // track indices in perm 
    memset(seen, 0, perm_ndim * sizeof(bool));  
    for (int i = 0; i < perm_ndim; i++) {
        int perm_val = perm[i]; 
        if (perm_val < 0 || perm_val >= perm_ndim) { // out of bounds
            return NULL; 
        } 
        if (seen[perm_val]) { // already covered
            return NULL;
        }
        seen[perm_val] = true; 
    }

    Tensor *new_t = tensor_view(t); 

    for (int i = 0; i < perm_ndim; i++) {
        new_t->shape[i] = t->shape[perm[i]];
        new_t->strides[i] = t->strides[perm[i]]; 
    }

    return new_t; 
};

Tensor *tensor_transpose(Tensor *t) {
    int perm[t->ndim];
    for (int i = t->ndim-1; i >= 0; i--) {
        perm[i] = t->ndim - 1 - i; 
    }
    return tensor_permute(t, perm, t->ndim); 
}


