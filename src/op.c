

#include <stdlib.h>
#include "../include/op.h"
#include "../include/tensor_ops.h"


Op *op_create(OpType type, Op *parent_1, Op *parent_2, int num_parents, OpAux aux) {
    Op *op = malloc(sizeof(Op)); 
    op->type = type; 
    op->parents[0] = parent_1; 
    op->parents[1] = parent_2; 
    op->num_parents = num_parents; 
    op->out = NULL; 
    op->aux = aux; 
    return op; 
};


void op_free() {

};






void forward(Op *op) {
    switch (op->type) {
        case ADD: op->out = elementwise_add(op->parents[0]->out, op->parents[1]->out); break; 
        case HADAMARD_PRODUCT: op->out = hadamard_product(op->parents[0]->out, op->parents[1]->out); break;  
        case MATRIX_PRODUCT: op->out = matrix_product(op->parents[0]->out, op->parents[1]->out); break; 
        
        case REDUCE_ADD: op->out = dim_reduce(op->parents[0]->out, op->aux.reduce.dim, ADD); break;   

        case RESHAPE: op->out = tensor_reshape(op->parents[0]->out, op->aux.reshape.new_shape, op->aux.reshape.new_ndim); break; 
        case BROADCAST: op->out = tensor_broadcast(op->parents[0]->out, op->aux.broadcast.dim, op->aux.broadcast.val); break; 
    }
};


void add_back(Op *op) {

};

void hadamard_product_back(Op *op) {

};

void matrix_product_back(Op *op) {

};

void reduce_add_back(Op *op) {

};


void backward(Op *op) {

}