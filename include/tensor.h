




typedef enum { 
    DTYPE_FP32, // only fp32 for now
} DType;


typedef struct { 
    float *data; // pointer to data 
    int *shape; // pointer to shape 
    int *strides; // pointer to stride
    int ndim;  // number of dims 
    int size;  // total num elements
    DType dtype; 
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
Tensor* tensor_create_constant(int value, int *shape, int ndim);
Tensor* tensor_create_rand(int *shape, int ndim);
void tensor_free(Tensor *t);
