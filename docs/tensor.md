# Tensor

## Struct

```c
typedef struct Tensor {
    float  *data;          // flat data buffer (row-major)
    int    *shape;         // [ndim]
    int    *strides;       // [ndim], in elements
    int     ndim;
    int     size;          // total element count
    DType   dtype;         // only DTYPE_FP32 exists
    bool    is_view;       // if true, data is not owned — tensor_free skips free(data)
    bool    requires_grad;
    struct Tensor *grad;   // same shape/size as self; grads never have grads (grad->grad == NULL)
} Tensor;
```

## Memory model

Tensors are either **owned** (`is_view = false`) or **views** (`is_view = true`).

- Views share the underlying `data` pointer with the source tensor. Modifying one modifies the other.
- `tensor_free` only calls `free(data)` for owned tensors.
- Views have their own `shape` and `strides` arrays (always heap-allocated and always freed).
- Broadcast tensors are views with `strides[dim] = 0` at the broadcast dimension.

## Strides

Row-major (C-order). For shape `[d0, d1, ..., dn]`:

```
strides[n]   = 1
strides[n-1] = shape[n]
strides[i]   = shape[i+1] * strides[i+1]
```

Element access: `data[dot(index, strides)]`

A stride of 0 at dimension `d` means every index along `d` maps to the same memory location — used to implement broadcast without copying data.

## Creation

```c
Tensor *tensor_create_constant(int value, int *shape, int ndim);
```
Allocates owned tensor filled with `value`. `requires_grad = false`, `grad = NULL`.

```c
Tensor *tensor_create_rand(int *shape, int ndim);
```
Same but filled with `rand() / RAND_MAX` values. Not seeded — call `srand` externally.

## Derived tensors

```c
Tensor *tensor_view(Tensor *t);
```
Shallow copy: shares `data`, copies `shape` and `strides` arrays. Returns `is_view = true`.

```c
Tensor *tensor_copy(Tensor *t);
```
Deep copy. If `t` is contiguous: `memcpy` data and strides. If not: materializes element-by-element in row-major order (resets strides to contiguous). Also deep-copies `grad` if present.

```c
Tensor *tensor_contiguous(Tensor *t);
```
Returns a contiguous copy. If `t` is already contiguous, delegates to `tensor_copy`. Otherwise materializes to a new row-major buffer.

## Shape operations

All shape ops return a new tensor (view or copy); they never modify in place.

```c
Tensor *tensor_reshape(Tensor *t, int *new_shape, int new_ndim);
```
Returns `NULL` if element count changes. If `t` is contiguous, returns a view with new shape/strides. Otherwise materializes a contiguous copy first.

```c
Tensor *tensor_broadcast(Tensor *t, int dim, int broadcast_val);
```
Returns `NULL` if `shape[dim] != 1`. Returns a view with `shape[dim] = broadcast_val` and `strides[dim] = 0`. Updates `size` accordingly.

```c
Tensor *tensor_permute(Tensor *t, int *perm, int perm_ndim);
```
Returns a view with axes reordered by `perm`. Returns `NULL` if `perm` is invalid (out-of-bounds indices or duplicates).

```c
Tensor *tensor_transpose(Tensor *t);
```
Full axis reversal via `tensor_permute`. Works for any `ndim`.

```c
Tensor *tensor_axis_transpose(Tensor *t);
```
Swaps only the last two axes. Requires `ndim == 2`; returns `NULL` otherwise. Prefer `tensor_transpose` for general use.

## Element access (inline)

```c
float tensor_get(Tensor *t, int *index);  // index is int[ndim]
void  tensor_set(Tensor *t, int *index, float value);
```
Both compute offset as `dot(index, strides)`. Safe to use on non-contiguous and broadcast tensors.

## Flat-to-multi-dim index

```c
void build_index(int *index, int flat_index, int *shape, int ndim);
```
Decomposes a flat index into a multi-dim index in row-major order. Used internally by reduction and copy loops. `index` must be pre-allocated as `int[ndim]`.

## Free

```c
void tensor_free(Tensor *t);
```
Frees `shape`, `strides`, and `grad` (recursively). Only frees `data` if `!is_view`.