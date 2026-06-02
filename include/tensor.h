




typedef enum { 
    DTYPE_FP32, // only fp32 for now
} DType;


typedef struct { 
    float *data; // pointer to data 
    int *shape; // pointer to shape 
    int ndim;  // number of dims 
    int size;  // total num elements
    DType dtype; 
} Tensor;