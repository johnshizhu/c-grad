#include <stdio.h>
#include "../include/tensor.h"

static int passed = 0;
static int failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); passed++; } \
    else       { printf("  FAIL: %s\n", msg); failed++; } \
} while(0)

void test_create_constant() {
    printf("test_create_constant\n");
    int shape[] = {2, 3};
    Tensor *t = tensor_create_constant(5, shape, 2);

    CHECK(t->ndim == 2,        "ndim is 2");
    CHECK(t->size == 6,        "size is 6");
    CHECK(t->shape[0] == 2,    "shape[0] is 2");
    CHECK(t->shape[1] == 3,    "shape[1] is 3");
    CHECK(t->is_view == false, "is_view is false");

    int idx[] = {0, 0};
    CHECK(tensor_get(t, idx) == 5.0f, "value at [0,0] is 5");
    idx[0] = 1; idx[1] = 2;
    CHECK(tensor_get(t, idx) == 5.0f, "value at [1,2] is 5");

    tensor_free(t);
}

void test_strides() {
    printf("test_strides\n");
    int shape[] = {2, 3};
    Tensor *t = tensor_create_constant(0, shape, 2);

    // row-major: shape {2,3} -> strides {3,1}
    CHECK(t->strides[0] == 3, "stride[0] is 3");
    CHECK(t->strides[1] == 1, "stride[1] is 1");

    tensor_free(t);
}

void test_view() {
    printf("test_view\n");
    int shape[] = {2, 3};
    Tensor *orig = tensor_create_constant(7, shape, 2);
    Tensor *view = tensor_view(orig);

    CHECK(view->is_view == true,    "view is_view is true");
    CHECK(view->data == orig->data, "view shares data pointer");
    CHECK(view->size == orig->size, "view size matches");

    tensor_free(view);
    tensor_free(orig);
}

void test_copy() {
    printf("test_copy\n");
    int shape[] = {2, 2};
    Tensor *orig = tensor_create_constant(3, shape, 2);
    Tensor *copy = tensor_copy(orig);

    CHECK(copy->is_view == false,   "copy is_view is false");
    CHECK(copy->data != orig->data, "copy has separate data pointer");

    int idx[] = {0, 0};
    CHECK(tensor_get(copy, idx) == 3.0f, "copy value matches original");

    tensor_free(copy);
    tensor_free(orig);
}

void test_reshape() {
    printf("test_reshape\n");
    int shape[] = {2, 3};
    Tensor *t = tensor_create_constant(1, shape, 2);

    int new_shape[] = {3, 2};
    Tensor *r = tensor_reshape(t, new_shape, 2);

    CHECK(r != NULL,        "reshape returns non-null");
    CHECK(r->ndim == 2,     "reshaped ndim is 2");
    CHECK(r->shape[0] == 3, "reshaped shape[0] is 3");
    CHECK(r->shape[1] == 2, "reshaped shape[1] is 2");
    CHECK(r->size == 6,     "reshaped size is 6");

    int bad_shape[] = {4, 2};
    Tensor *bad = tensor_reshape(t, bad_shape, 2);
    CHECK(bad == NULL, "reshape with wrong size returns NULL");

    tensor_free(r);
    tensor_free(t);
}

void test_broadcast() {
    printf("test_broadcast\n");
    int shape[] = {1, 3};
    Tensor *t = tensor_create_constant(2, shape, 2);
    Tensor *b = tensor_broadcast(t, 0, 4);

    CHECK(b != NULL,          "broadcast returns non-null");
    CHECK(b->shape[0] == 4,   "broadcast shape[0] is 4");
    CHECK(b->strides[0] == 0, "broadcast stride[0] is 0");

    tensor_free(b);
    tensor_free(t);
}

int main() {
    test_create_constant();
    test_strides();
    test_view();
    test_copy();
    test_reshape();
    test_broadcast();

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
