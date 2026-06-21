# Tensor Ops

All ops in `src/tensor_ops.c` / `include/tensor_ops.h`. They operate directly on `Tensor*` values and return a newly allocated owned tensor. They do **not** touch the computation graph — graph tracking is the responsibility of `Op` (see [ops.md](ops.md)).

## Shape requirement

```c
bool check_shape_match(Tensor *t1, Tensor *t2);  // internal
```
Requires identical `ndim`, `size`, and per-dimension `shape`. Used as a precondition guard; returns `NULL` from callers on mismatch.

## Elementwise ops

```c
Tensor *elementwise_add(Tensor *t1, Tensor *t2);
Tensor *hadamard_product(Tensor *t1, Tensor *t2);
```
Both require exact shape match (no implicit broadcasting — broadcast manually first). Iterate in row-major flat order via `build_index` + `tensor_get/set`, so they handle non-contiguous inputs correctly.

## Reduction

```c
typedef enum { SUM, MAX, MEAN } ReduceOp;

Tensor *dim_reduce(Tensor *t, int dim, ReduceOp op);
```

Reduces along `dim`. Output shape is `t->shape` with `dim` removed (`ndim - 1`).

| `op`   | behavior                                           |
|--------|----------------------------------------------------|
| `SUM`  | accumulate sum                                     |
| `MAX`  | running max initialized to first element           |
| `MEAN` | accumulate sum, then divide by `t->shape[dim]`     |

Implementation: iterates over all output positions, maps each back into input index space by injecting `dim`, then loops over `k = 0..shape[dim]-1`.

## Matrix product

```c
Tensor *matrix_product(Tensor *t1, Tensor *t2);
```

Tinygrad-style decomposition — no explicit triple loop:

1. **Unsqueeze** `t1` at trailing dim → shape `[..., M, K, 1]`
2. **Unsqueeze** `t2` at leading dim → shape `[1, ..., K, N]`  *(note: prepended)*
3. **Broadcast** `t1` trailing dim to `N` → `[..., M, K, N]`
4. **Broadcast** `t2` leading dim to `M` → `[M, ..., K, N]`  *(wait — see below)*
5. **Hadamard product** — element-wise multiply `[..., M, K, N]`
6. **`dim_reduce(..., ndim-2, SUM)`** — sum along `K` → `[..., M, N]`

Validity check (`check_mul_compatible`): requires `ndim >= 2` for both, `t1->shape[ndim-1] == t2->shape[ndim-2]` (K matches), `ndim` equal, and all batch dims identical (no batch broadcasting).

Returns `NULL` on incompatible shapes.

> **Note on the unsqueeze placement:** `t2_new_shape[0] = 1` prepends a size-1 dimension; broadcasting then expands it to match `t1`'s leading batch/row dimension. The implementation currently broadcasts at index `0` of `t2_r` to `t1_r->shape[0]`, which is correct for 2D but may not generalize cleanly to batched inputs without revisiting the dim arithmetic.