# c-grad

Simple tensor library with autograd in C.

Op-centric design: `Op` nodes form a DAG and own their output tensors. Tensors are pure value objects (shape, strides, data, optional grad). Autograd is driven by explicit `forward()` / `backward()` calls on op nodes.

## Docs

- [docs/tensor.md](docs/tensor.md) — `Tensor` struct, memory model, shape ops
- [docs/tensor_ops.md](docs/tensor_ops.md) — elementwise, reduction, matrix product
- [docs/ops.md](docs/ops.md) — computation graph, `Op` struct, forward/backward

## Build

```sh
make          # compiles all tests into build/
make clean    # removes build/
```

## Status

| Component      | State                                      |
|----------------|--------------------------------------------|
| `Tensor`       | Complete — creation, views, reshape, broadcast, permute, copy |
| `tensor_ops`   | Complete — add, hadamard, dim_reduce (sum/max/mean), matrix_product |
| `Op` / graph   | Forward and per-op backward implemented; graph-level auto-traversal planned |
| `nn`           | Stub only                                  |