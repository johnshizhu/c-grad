




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

// prototypes
Tensor* tensor_create_constant(int value, int *shape, int ndim);
Tensor* tensor_create_rand(int *shape, int ndim);
void tensor_free(Tensor *t);
