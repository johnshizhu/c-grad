# Computation Graph & Autograd

## Design

The graph is **Op-centric**. Each `Op` node owns its output `Tensor` and holds pointers to its parent `Op` nodes (not to tensors directly). Tensors are pure value objects — they carry no pointer back to the op that created them. Gradient accumulation is driven by `backward(Op*)`, which writes into parent tensors' `grad` fields.

```
Op { type, parents[2], out }
         |                \
    Op (parent 0)     Op (parent 1)
         |
       ...
```

Leaf nodes (`num_parents == 0`) hold raw input tensors in `out`.

## Structs

```c
typedef enum {
    ADD,
    HADAMARD_PRODUCT,
    REDUCE_ADD,
    RESHAPE,
    BROADCAST,
} OpType;

typedef union {
    struct { int dim; }                    reduce;    // REDUCE_ADD
    struct { int *new_shape; int new_ndim; } reshape; // RESHAPE
    struct { int dim; int val; }           broadcast; // BROADCAST
} OpAux;

typedef struct Op {
    OpType   type;
    struct Op *parents[2];
    int       num_parents;
    Tensor   *out;     // result of forward(); NULL before forward() is called
    OpAux     aux;
} Op;
```

`OpAux` carries the extra parameters that can't be inferred from the input shapes alone. `RESHAPE` heap-allocates `new_shape` — `op_free` owns that memory.

## Lifecycle

```c
Op *op_create(OpType type, Op *parent_1, Op *parent_2, int num_parents, OpAux aux);
void op_free(Op *op);
```

`op_create` allocates and wires parents; `out` is set to `NULL` until `forward()` runs.  
`op_free` frees `out` (via `tensor_free`) and, for `RESHAPE`, frees `aux.reshape.new_shape`.  
It does **not** recursively free parent ops — graph teardown order is the caller's responsibility.

## Forward pass

```c
void forward(Op *op);
```

Dispatches to the corresponding tensor op and writes the result into `op->out`.

| `OpType`          | calls                                                           |
|-------------------|-----------------------------------------------------------------|
| `ADD`             | `elementwise_add(parents[0]->out, parents[1]->out)`             |
| `HADAMARD_PRODUCT`| `hadamard_product(parents[0]->out, parents[1]->out)`            |
| `REDUCE_ADD`      | `dim_reduce(parents[0]->out, aux.reduce.dim, SUM)`              |
| `RESHAPE`         | `tensor_reshape(parents[0]->out, aux.reshape.new_shape, ...)`   |
| `BROADCAST`       | `tensor_broadcast(parents[0]->out, aux.broadcast.dim, aux.broadcast.val)` |

Must be called in topological order (parents before children).

## Backward pass

```c
void backward(Op *op);
```

Assumes `op->out->grad` has already been set (the upstream gradient). Computes and accumulates gradients into each parent's `out->grad`.

Gradient accumulation helper:

```c
void accum_grad(Tensor **target, Tensor *grad);
```

If `*target == NULL`, copies `grad` into a new tensor. Otherwise elementwise-adds `grad` into `*target` and frees the old tensor. This handles the case where a tensor is used by multiple ops (fan-out in the graph).

### Per-op gradient rules

**ADD**
```
∂L/∂p0 += out->grad
∂L/∂p1 += out->grad
```

**HADAMARD_PRODUCT**
```
∂L/∂p0 += p1->out ⊙ out->grad
∂L/∂p1 += p0->out ⊙ out->grad
```

**REDUCE_ADD** (reduces along `dim`)
```
grad_unsqueeze = reshape(out->grad, insert size-1 at dim)
grad_broad     = broadcast(grad_unsqueeze, dim, p0->shape[dim])
∂L/∂p0        += contiguous(grad_broad)
```
The unsqueeze + broadcast restores the reduced dimension so the gradient matches the input shape.

**RESHAPE**
```
∂L/∂p0 += reshape(out->grad, p0->out->shape)
```

**BROADCAST** (broadcasts along `dim`)
```
grad_red = dim_reduce(out->grad, dim, SUM)
∂L/∂p0  += reshape(grad_red, p0->out->shape)   // re-inserts the size-1 dim
```
Summing along the broadcast dimension reverses the replication.

Must be called in **reverse** topological order (children before parents). The root op's `out->grad` must be seeded before calling `backward` on any node (typically a ones tensor of the same shape as the output).

## Planned: graph-level backward

Currently `backward(Op*)` must be called manually per node. Planned addition: a `backward_graph(Op *root)` that:

1. Performs a topological sort of the DAG rooted at `root`
2. Seeds `root->out->grad` with a ones tensor
3. Calls `backward` on each node in reverse topological order

## Building a computation graph (example)

```c
// Leaf ops (no parents) hold input tensors
OpAux no_aux = {0};
Op *a = op_create(ADD, NULL, NULL, 0, no_aux);  // placeholder type, 0 parents
a->out = tensor_create_constant(2, shape, ndim);

Op *b = op_create(ADD, NULL, NULL, 0, no_aux);
b->out = tensor_create_constant(3, shape, ndim);

// Intermediate op
Op *add_op = op_create(ADD, a, b, 2, no_aux);
forward(add_op);   // add_op->out = a->out + b->out

// Seed gradient at output, then backprop
add_op->out->grad = tensor_create_constant(1, add_op->out->shape, add_op->out->ndim);
backward(add_op);  // accumulates into a->out->grad and b->out->grad
```