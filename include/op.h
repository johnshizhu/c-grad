typedef struct Tensor Tensor; 

// Node in computation graphfor autograd


typedef enum { 
    ADD,
    MATRIX_PRODUCT, 
    REDUCE_ADD, 
} OpType;


typedef struct { 
    OpType type; 
    struct Op *parents[2]; // at max 2 inputs
    int num_parents; 
    Tensor *out; // result of the Op reading in the result of the parents
} Op; 

// in the case that num_parents = 0, set the out field as raw tensor values






