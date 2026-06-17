#include <stdbool.h>
#include <../include/op.h> 


typedef enum { 
    DTYPE_FP32, // only fp32 for now
} DType;


typedef struct Tensor {
    float *data;
    int *shape;
    int *strides;
    int ndim;
    int size;
    DType dtype;
    bool is_view;
    bool requires_grad;
    struct Tensor *grad;
} Tensor;


static inline float tensor_get(Tensor *t, int *index) {
    int offset = 0; // offset is the dot product of the index and strides vectors
    for (int dim = 0; dim < t->ndim; dim++) {
        offset = offset + (index[dim] * t->strides[dim]);
    }

    return t->data[offset];
};


static inline void tensor_set(Tensor *t, int *index, float value) {
    int offset = 0; // offset is the dot product of the index and strides vectors
    for (int dim = 0; dim < t->ndim; dim++) {
        offset = offset + (index[dim] * t->strides[dim]);
    }
    t->data[offset] = value; 
};


// prototypes
Tensor *tensor_create_constant(int value, int *shape, int ndim);
Tensor *tensor_create_rand(int *shape, int ndim);
Tensor *tensor_view(Tensor *t); 
Tensor *tensor_copy(Tensor *t); 
Tensor *tensor_broadcast(Tensor *t, int dim, int broadcast_val); 
Tensor *tensor_reshape(Tensor *t, int *new_shape, int new_dim);
Tensor *tensor_axis_transpose(Tensor *t); 
Tensor *tensor_permute(Tensor *t, int *perm, int perm_ndim); 
Tensor *tensor_transpose(Tensor *t); 
void tensor_free(Tensor *t);
