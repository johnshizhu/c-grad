

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
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

void op_free(Op *op) {
    if (op->out != NULL) {
        tensor_free(op->out); 
    }
    if (op->type == RESHAPE) {
        free(op->aux.reshape.new_shape);
    }
    free(op); 
};


void forward(Op *op) {
    switch (op->type) {
        case ADD: op->out = elementwise_add(op->parents[0]->out, op->parents[1]->out); break; 
        case HADAMARD_PRODUCT: op->out = hadamard_product(op->parents[0]->out, op->parents[1]->out); break;  
        
        case REDUCE_ADD: op->out = dim_reduce(op->parents[0]->out, op->aux.reduce.dim, SUM); break;   

        case RESHAPE: op->out = tensor_reshape(op->parents[0]->out, op->aux.reshape.new_shape, op->aux.reshape.new_ndim); break; 
        case BROADCAST: op->out = tensor_broadcast(op->parents[0]->out, op->aux.broadcast.dim, op->aux.broadcast.val); break; 
    }
};


void accum_grad(Tensor **target, Tensor *grad) {
    if (*target == NULL) { // new inialization
        *target = tensor_copy(grad); 
    }
    else { // accumulation 
        Tensor *tmp = *target; 
        *target = elementwise_add(*target, grad); 
        tensor_free(tmp); 
    }
}

void add_back(Op *op) { // grad is 1
    accum_grad(&op->parents[0]->out->grad, op->out->grad); 
    accum_grad(&op->parents[1]->out->grad, op->out->grad); 
};

void hadamard_product_back(Op *op) {
    Tensor *p0_grad = hadamard_product(op->parents[1]->out, op->out->grad); 
    Tensor *p1_grad = hadamard_product(op->parents[0]->out, op->out->grad); 
    accum_grad(&op->parents[0]->out->grad, p0_grad); 
    accum_grad(&op->parents[1]->out->grad, p1_grad); 
    tensor_free(p0_grad); 
    tensor_free(p1_grad); 
};

void reduce_add_back(Op *op) {
    int dim = op->aux.reduce.dim; 
    int *new_shape = malloc((op->out->ndim + 1) * sizeof(int));
    memcpy(new_shape, op->out->grad->shape, dim * sizeof(int)); 
    new_shape[dim] = 1; 
    memcpy(new_shape + dim + 1, op->out->grad->shape + dim, (op->out->ndim - dim) * sizeof(int));     
    Tensor *grad_unsqueeze = tensor_reshape(op->out->grad, new_shape, op->out->ndim + 1); 
    Tensor *grad_broad = tensor_broadcast(grad_unsqueeze, dim, op->parents[0]->out->shape[dim]);
    Tensor *grad_cont = tensor_contiguous(grad_broad); 
    accum_grad(&op->parents[0]->out->grad, grad_cont); 
    free(new_shape); 
    tensor_free(grad_unsqueeze); 
    tensor_free(grad_broad); 
    tensor_free(grad_cont); 
};

void reshape_back(Op *op) {
    Tensor *grad_r = tensor_reshape(op->out->grad, op->parents[0]->out->shape, op->parents[0]->out->ndim);
    accum_grad(&op->parents[0]->out->grad, grad_r); 
    tensor_free(grad_r); 
};

void broadcast_back(Op *op) {
    int dim = op->aux.broadcast.dim; 
    Tensor *grad_red = dim_reduce(op->out->grad, dim, SUM); 
    int *new_shape = malloc((grad_red->ndim + 1) * sizeof(int)); 
    memcpy(new_shape, op->out->grad->shape, dim * sizeof(int)); 
    new_shape[dim] = 1; 
    memcpy(new_shape + dim + 1, op->out->grad->shape + dim + 1, (op->out->ndim - dim - 1) * sizeof(int));
    Tensor *grad_r = tensor_reshape(grad_red, new_shape, op->parents[0]->out->ndim); 
    accum_grad(&op->parents[0]->out->grad, grad_r); 
    free(new_shape); 
    tensor_free(grad_red); 
    tensor_free(grad_r); 
};


void backward(Op *op) {
    switch (op->type) {
        case ADD: add_back(op); break; 
        case HADAMARD_PRODUCT: hadamard_product_back(op); break; 
        
        case REDUCE_ADD: reduce_add_back(op); break; 
        
        case RESHAPE: reshape_back(op); break; 
        case BROADCAST: broadcast_back(op); break; 
    }

}