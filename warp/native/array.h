// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

namespace wp {

#if FP_CHECK

#define FP_ASSERT_FWD(value) \
    print(value); \
    printf(")\n"); \
    assert(0);

#define FP_ASSERT_ADJ(value, adj_value) \
    print(value); \
    printf(", "); \
    print(adj_value); \
    printf(")\n"); \
    assert(0);

#define FP_VERIFY_FWD(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(addr", __FILE__, __LINE__, __FUNCTION__); \
        FP_ASSERT_FWD(value) \
    }

#define FP_VERIFY_FWD_1(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d) ", __FILE__, __LINE__, __FUNCTION__, i); \
        FP_ASSERT_FWD(value) \
    }

#define FP_VERIFY_FWD_2(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j); \
        FP_ASSERT_FWD(value) \
    }

#define FP_VERIFY_FWD_3(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k); \
        FP_ASSERT_FWD(value) \
    }

#define FP_VERIFY_FWD_4(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k, l); \
        FP_ASSERT_FWD(value) \
    }

#define FP_VERIFY_ADJ(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(addr",  __FILE__, __LINE__, __FUNCTION__); \
        FP_ASSERT_ADJ(value, adj_value); \
    }

#define FP_VERIFY_ADJ_1(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d) ",  __FILE__, __LINE__, __FUNCTION__, i); \
        FP_ASSERT_ADJ(value, adj_value); \
    }

#define FP_VERIFY_ADJ_2(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d, %d) ",  __FILE__, __LINE__, __FUNCTION__, i, j); \
        FP_ASSERT_ADJ(value, adj_value); \
    }

#define FP_VERIFY_ADJ_3(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k); \
        FP_ASSERT_ADJ(value, adj_value); \
    }

#define FP_VERIFY_ADJ_4(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k, l); \
        FP_ASSERT_ADJ(value, adj_value); \
    }


#else

#define FP_VERIFY_FWD(value) {}
#define FP_VERIFY_FWD_1(value) {}
#define FP_VERIFY_FWD_2(value) {}
#define FP_VERIFY_FWD_3(value) {}
#define FP_VERIFY_FWD_4(value) {}

#define FP_VERIFY_ADJ(value, adj_value) {}
#define FP_VERIFY_ADJ_1(value, adj_value) {}
#define FP_VERIFY_ADJ_2(value, adj_value) {}
#define FP_VERIFY_ADJ_3(value, adj_value) {}
#define FP_VERIFY_ADJ_4(value, adj_value) {}

#endif  // WP_FP_CHECK


template <size_t... Is> struct index_sequence { };

template <size_t N, size_t... Is> struct make_index_sequence_impl : make_index_sequence_impl<N - 1, N - 1, Is...> { };

template <size_t... Is> struct make_index_sequence_impl<0, Is...> {
    using type = index_sequence<Is...>;
};

template <size_t N> using make_index_sequence = typename make_index_sequence_impl<N>::type;


const int ARRAY_MAX_DIMS = 4;  // must match constant in types.py

// must match constants in types.py
const int ARRAY_TYPE_REGULAR = 0;
const int ARRAY_TYPE_INDEXED = 1;
const int ARRAY_TYPE_FABRIC = 2;
const int ARRAY_TYPE_FABRIC_INDEXED = 3;

constexpr uint16_t ARRAY_FLAG_RETAIN_GRAD = 1 << 0;

struct shape_t {
    int dims[ARRAY_MAX_DIMS];

    CUDA_CALLABLE constexpr shape_t(int d0 = 0, int d1 = 0, int d2 = 0, int d3 = 0)
        : dims { d0, d1, d2, d3 }
    {
    }

    CUDA_CALLABLE inline int operator[](int i) const
    {
        assert(i < ARRAY_MAX_DIMS);
        return dims[i];
    }

    CUDA_CALLABLE inline int& operator[](int i)
    {
        assert(i < ARRAY_MAX_DIMS);
        return dims[i];
    }
};

CUDA_CALLABLE inline int extract(const shape_t& s, int i) { return s.dims[i]; }

CUDA_CALLABLE inline void adj_extract(const shape_t& s, int i, const shape_t& adj_s, int adj_i, int adj_ret) { }

// Templated runtime-K shape access for kernel-specialize codegen's
// baked arrays.  Used when ``arr.shape[k]`` is hit with a non-literal
// ``k`` (rare — happens when the index is from a non-unrollable loop).
// Template args carry the static shape values; the runtime ternary
// collapses to a compare-select chain over compile-time constants
// without any field reads through a struct.
template <int S0, int S1, int S2, int S3>
CUDA_CALLABLE constexpr int baked_shape_extract(int k)
{
    return (k == 0) ? S0 : (k == 1) ? S1 : (k == 2) ? S2 : S3;
}

inline CUDA_CALLABLE void print(shape_t s)
{
    // todo: only print valid dims, currently shape has a fixed size
    // but we don't know how many dims are valid (e.g.: 1d, 2d, etc)
    // should probably store ndim with shape
    printf("(%d, %d, %d, %d)\n", s.dims[0], s.dims[1], s.dims[2], s.dims[3]);
}
inline CUDA_CALLABLE void adj_print(shape_t s, shape_t& adj_s) { }


template <typename T> struct array_t {
    CUDA_CALLABLE inline array_t()
        : data(nullptr)
        , grad(nullptr)
        , shape()
        , strides()
        , ndim(0)
        , flags(0)
    {
    }

    CUDA_CALLABLE array_t(T* data, int size, T* grad = nullptr)
        : data(data)
        , grad(grad)
    {
        // constructor for 1d array
        shape.dims[0] = size;
        shape.dims[1] = 0;
        shape.dims[2] = 0;
        shape.dims[3] = 0;
        ndim = 1;
        flags = 0;
        strides[0] = sizeof(T);
        strides[1] = 0;
        strides[2] = 0;
        strides[3] = 0;
    }
    CUDA_CALLABLE array_t(T* data, int dim0, int dim1, T* grad = nullptr)
        : data(data)
        , grad(grad)
    {
        // constructor for 2d array
        shape.dims[0] = dim0;
        shape.dims[1] = dim1;
        shape.dims[2] = 0;
        shape.dims[3] = 0;
        ndim = 2;
        flags = 0;
        strides[0] = dim1 * sizeof(T);
        strides[1] = sizeof(T);
        strides[2] = 0;
        strides[3] = 0;
    }
    CUDA_CALLABLE array_t(T* data, int dim0, int dim1, int dim2, T* grad = nullptr)
        : data(data)
        , grad(grad)
    {
        // constructor for 3d array
        shape.dims[0] = dim0;
        shape.dims[1] = dim1;
        shape.dims[2] = dim2;
        shape.dims[3] = 0;
        ndim = 3;
        flags = 0;
        strides[0] = dim1 * dim2 * sizeof(T);
        strides[1] = dim2 * sizeof(T);
        strides[2] = sizeof(T);
        strides[3] = 0;
    }
    CUDA_CALLABLE array_t(T* data, int dim0, int dim1, int dim2, int dim3, T* grad = nullptr)
        : data(data)
        , grad(grad)
    {
        // constructor for 4d array
        shape.dims[0] = dim0;
        shape.dims[1] = dim1;
        shape.dims[2] = dim2;
        shape.dims[3] = dim3;
        ndim = 4;
        flags = 0;
        strides[0] = dim1 * dim2 * dim3 * sizeof(T);
        strides[1] = dim2 * dim3 * sizeof(T);
        strides[2] = dim3 * sizeof(T);
        strides[3] = sizeof(T);
    }

    CUDA_CALLABLE array_t(uint64 data, int size, uint64 grad = 0)
        : array_t((T*)(data), size, (T*)(grad))
    {
    }

    CUDA_CALLABLE array_t(uint64 data, int dim0, int dim1, uint64 grad = 0)
        : array_t((T*)(data), dim0, dim1, (T*)(grad))
    {
    }

    CUDA_CALLABLE array_t(uint64 data, int dim0, int dim1, int dim2, uint64 grad = 0)
        : array_t((T*)(data), dim0, dim1, dim2, (T*)(grad))
    {
    }

    CUDA_CALLABLE array_t(uint64 data, int dim0, int dim1, int dim2, int dim3, uint64 grad = 0)
        : array_t((T*)(data), dim0, dim1, dim2, dim3, (T*)(grad))
    {
    }

    CUDA_CALLABLE inline bool empty() const { return !data; }

    T* data;
    T* grad;
    shape_t shape;
    int strides[ARRAY_MAX_DIMS];
    uint16_t ndim;
    uint16_t flags;

    CUDA_CALLABLE inline operator T*() const { return data; }
};


// Compile-time array proxy used by the kernel-specialize codegen for
// the "baked array passed by value" pattern (`wp::view(arr, ...)`,
// `wp::where(arr, a, b)`, `if (arr)`).  Inherits from `array_t<T>` so:
//
//   - All consumers of `array_t<T>&` (`tile_load`, `tile_store`,
//     the variadic-slice `wp::view`, indexedarray, ...) accept a
//     baked_array_t via base-class upcast at no runtime cost.  No
//     conversion operator, no `array_t<T> tmp = src;` materialisation
//     in the kernel body — the upcast is purely a type change.
//   - The inherited shape / strides / ndim / flags fields are
//     written exactly once at the baked_array_t ctor, from the
//     compile-time template args.  NVRTC folds those writes the
//     same way it does for the pre-Phase-F direct-field reconstruction
//     pattern (Phase E baseline IR), and downstream `src.shape[i]` /
//     `src.strides[i]` reads inside generic consumers fold to
//     constants.
//
// Templated overloads of `view` below fire ahead of the generic
// `array_t<T>&` versions when the static baked type is preserved at
// the call site.  Those overloads bypass the inherited fields entirely
// and read shape / stride from the template args directly, which is
// what the user sees as "no constructed shape_t" for the int-indexed
// view path.
template <typename T, int Ndim, int S0, int S1, int S2, int S3, int St0, int St1, int St2, int St3>
struct baked_array_t : array_t<T> {
    // Static accessors exposing the template args as constexpr
    // values so generic templated consumers (e.g. `tile_global_t`,
    // `is_baked_array_t` traits) can route shape / stride reads
    // through compile-time constants without an SFINAE/specialisation
    // dance.  Reads of `Src::baked_strides[i]` with a constexpr `i`
    // (after WP_PRAGMA_UNROLL) collapse to immediates at ptxas, so
    // tile_global_t::index() no longer depends on NVRTC dataflow
    // analysis to fold the inherited `data.strides[i]` field reads.
    static constexpr int baked_ndim = Ndim;
    static constexpr int baked_shape[ARRAY_MAX_DIMS] = { S0, S1, S2, S3 };
    static constexpr int baked_strides[ARRAY_MAX_DIMS] = { St0, St1, St2, St3 };

    // Default ctor: leaves inherited array_t<T> default-initialized
    // (data=nullptr, fields zero).  Used when codegen declares a
    // view-result local as `baked_array_t<...> var_X;` and assigns
    // the return value of `wp::view(...)` later — the write-once
    // assignment overwrites all fields, NVRTC eliminates the dead
    // zero-init writes.  Lets the local carry its static type
    // through downstream consumers (tile_load, etc.) instead of
    // being slice-assigned to a plain `array_t<T>`.
    CUDA_CALLABLE baked_array_t() = default;

    // Single-arg ctor: take the runtime data pointer; populate the
    // inherited array_t<T> fields from the compile-time template args.
    // The writes are constexpr-known values, so NVRTC folds them
    // away and any subsequent `src.shape[i]` / `src.strides[i]` /
    // `src.ndim` read in a generic consumer collapses to a literal.
    CUDA_CALLABLE baked_array_t(T* data_ptr)
    {
        this->data = data_ptr;
        this->grad = nullptr;
        this->shape.dims[0] = S0;
        this->shape.dims[1] = S1;
        this->shape.dims[2] = S2;
        this->shape.dims[3] = S3;
        this->strides[0] = St0;
        this->strides[1] = St1;
        this->strides[2] = St2;
        this->strides[3] = St3;
        this->ndim = Ndim;
        this->flags = 0;
    }
};


// Compile-time stride / shape lookup for a `baked_array_t<...>`.  The
// ternary chain returns one of the template-arg values selected by a
// runtime `i`; when called from inside a `WP_PRAGMA_UNROLL`'d loop the
// `i` becomes a literal at each unrolled iteration and the ternary
// collapses to the matching constant — no memory load, no dataflow-
// analysis dependency.  Used by `tile_global_t::index` to drive its
// offset math from the source's static template args when Src is
// baked.  The generic `array_t<T>&` fallback below is selected when
// the source isn't baked.
template <typename T, int Ndim, int S0, int S1, int S2, int S3, int St0, int St1, int St2, int St3>
CUDA_CALLABLE inline int tile_strides_at(const baked_array_t<T, Ndim, S0, S1, S2, S3, St0, St1, St2, St3>&, int i)
{
    return (i == 0) ? St0 : (i == 1) ? St1 : (i == 2) ? St2 : St3;
}

template <typename T> CUDA_CALLABLE inline int tile_strides_at(const array_t<T>& src, int i) { return src.strides[i]; }

template <typename T, int Ndim, int S0, int S1, int S2, int S3, int St0, int St1, int St2, int St3>
CUDA_CALLABLE inline int tile_shape_at(const baked_array_t<T, Ndim, S0, S1, S2, S3, St0, St1, St2, St3>&, int i)
{
    return (i == 0) ? S0 : (i == 1) ? S1 : (i == 2) ? S2 : S3;
}

template <typename T> CUDA_CALLABLE inline int tile_shape_at(const array_t<T>& src, int i) { return src.shape[i]; }


// Required when compiling adjoints.
template <typename T> inline CUDA_CALLABLE array_t<T> add(const array_t<T>& a, const array_t<T>& b)
{
    return array_t<T>();
}


// Stack‑allocated counterpart to `array_t<T>`.
// Useful for small buffers that have their shape known at compile-time,
// and that gain from having array semantics instead of vectors.
template <int Size, typename T> struct fixedarray_t : array_t<T> {
    using Base = array_t<T>;

    static_assert(Size > 0, "Expected Size > 0");

    CUDA_CALLABLE inline fixedarray_t()
        : Base(storage, Size)
        , storage()
    {
    }

    CUDA_CALLABLE fixedarray_t(int dim0, T* grad = nullptr)
        : Base(storage, dim0, grad)
        , storage()
    {
        assert(Size == dim0);
    }

    CUDA_CALLABLE fixedarray_t(int dim0, int dim1, T* grad = nullptr)
        : Base(storage, dim0, dim1, grad)
        , storage()
    {
        assert(Size == dim0 * dim1);
    }

    CUDA_CALLABLE fixedarray_t(int dim0, int dim1, int dim2, T* grad = nullptr)
        : Base(storage, dim0, dim1, dim2, grad)
        , storage()
    {
        assert(Size == dim0 * dim1 * dim2);
    }

    CUDA_CALLABLE fixedarray_t(int dim0, int dim1, int dim2, int dim3, T* grad = nullptr)
        : Base(storage, dim0, dim1, dim2, dim3, grad)
        , storage()
    {
        assert(Size == dim0 * dim1 * dim2 * dim3);
    }

    CUDA_CALLABLE fixedarray_t<Size, T>& operator=(const fixedarray_t<Size, T>& other)
    {
        for (unsigned int i = 0; i < Size; ++i) {
            this->storage[i] = other.storage[i];
        }

        this->data = this->storage;
        this->grad = nullptr;
        this->shape = other.shape;

        for (unsigned int i = 0; i < ARRAY_MAX_DIMS; ++i) {
            this->strides[i] = other.strides[i];
        }

        this->ndim = other.ndim;
        this->flags = other.flags;

        return *this;
    }

    T storage[Size];
};


// Required when compiling adjoints.
template <int Size, typename T>
inline CUDA_CALLABLE fixedarray_t<Size, T> add(const fixedarray_t<Size, T>& a, const fixedarray_t<Size, T>& b)
{
    return fixedarray_t<Size, T>();
}


// TODO:
// - templated index type?
// - templated dimensionality? (also for array_t to save space when passing arrays to kernels)
template <typename T> struct indexedarray_t {
    CUDA_CALLABLE inline indexedarray_t()
        : arr()
        , indices()
        , shape()
    {
    }

    CUDA_CALLABLE inline bool empty() const { return !arr.data; }

    array_t<T> arr;
    int* indices[ARRAY_MAX_DIMS];  // index array per dimension (can be NULL)
    shape_t shape;  // element count per dimension (num. indices if indexed, array dim if not)
};


// return stride (in bytes) of the given index
template <typename T> CUDA_CALLABLE inline size_t stride(const array_t<T>& a, int dim)
{
    return size_t(a.strides[dim]);
}

template <typename T> CUDA_CALLABLE inline T* data_at_byte_offset(const array_t<T>& a, size_t byte_offset)
{
    return reinterpret_cast<T*>(reinterpret_cast<char*>(a.data) + byte_offset);
}

template <typename T> CUDA_CALLABLE inline T* grad_at_byte_offset(const array_t<T>& a, size_t byte_offset)
{
    return reinterpret_cast<T*>(reinterpret_cast<char*>(a.grad) + byte_offset);
}

template <typename T> CUDA_CALLABLE inline size_t byte_offset(const array_t<T>& arr, int i)
{
    assert(i >= 0 && i < arr.shape[0]);

    return i * stride(arr, 0);
}

template <typename T> CUDA_CALLABLE inline size_t byte_offset(const array_t<T>& arr, int i, int j)
{
    // if (i < 0 || i >= arr.shape[0])
    //     printf("i: %d > arr.shape[0]: %d\n", i, arr.shape[0]);

    // if (j < 0 || j >= arr.shape[1])
    //     printf("j: %d > arr.shape[1]: %d\n", j, arr.shape[1]);


    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);

    return i * stride(arr, 0) + j * stride(arr, 1);
}

template <typename T> CUDA_CALLABLE inline size_t byte_offset(const array_t<T>& arr, int i, int j, int k)
{
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);
    assert(k >= 0 && k < arr.shape[2]);

    return i * stride(arr, 0) + j * stride(arr, 1) + k * stride(arr, 2);
}

template <typename T> CUDA_CALLABLE inline size_t byte_offset(const array_t<T>& arr, int i, int j, int k, int l)
{
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);
    assert(k >= 0 && k < arr.shape[2]);
    assert(l >= 0 && l < arr.shape[3]);

    return i * stride(arr, 0) + j * stride(arr, 1) + k * stride(arr, 2) + l * stride(arr, 3);
}

template <typename T> CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i)
{
    assert(arr.ndim == 1);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);

    if (i < 0) {
        i += arr.shape[0];
    }

    T& result = *data_at_byte_offset(arr, byte_offset(arr, i));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j)
{
    assert(arr.ndim == 2);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }

    T& result = *data_at_byte_offset(arr, byte_offset(arr, i, j));
    FP_VERIFY_FWD_2(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j, int k)
{
    assert(arr.ndim == 3);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);
    assert(k >= -arr.shape[2] && k < arr.shape[2]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }
    if (k < 0) {
        k += arr.shape[2];
    }

    T& result = *data_at_byte_offset(arr, byte_offset(arr, i, j, k));
    FP_VERIFY_FWD_3(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T& index(const array_t<T>& arr, int i, int j, int k, int l)
{
    assert(arr.ndim == 4);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);
    assert(k >= -arr.shape[2] && k < arr.shape[2]);
    assert(l >= -arr.shape[3] && l < arr.shape[3]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }
    if (k < 0) {
        k += arr.shape[2];
    }
    if (l < 0) {
        l += arr.shape[3];
    }

    T& result = *data_at_byte_offset(arr, byte_offset(arr, i, j, k, l));
    FP_VERIFY_FWD_4(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T& index_grad(const array_t<T>& arr, int i)
{
    assert(arr.ndim == 1);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);

    if (i < 0) {
        i += arr.shape[0];
    }

    T& result = *grad_at_byte_offset(arr, byte_offset(arr, i));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T& index_grad(const array_t<T>& arr, int i, int j)
{
    assert(arr.ndim == 2);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }

    T& result = *grad_at_byte_offset(arr, byte_offset(arr, i, j));
    FP_VERIFY_FWD_2(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T& index_grad(const array_t<T>& arr, int i, int j, int k)
{
    assert(arr.ndim == 3);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);
    assert(k >= -arr.shape[2] && k < arr.shape[2]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }
    if (k < 0) {
        k += arr.shape[2];
    }

    T& result = *grad_at_byte_offset(arr, byte_offset(arr, i, j, k));
    FP_VERIFY_FWD_3(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T& index_grad(const array_t<T>& arr, int i, int j, int k, int l)
{
    assert(arr.ndim == 4);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);
    assert(k >= -arr.shape[2] && k < arr.shape[2]);
    assert(l >= -arr.shape[3] && l < arr.shape[3]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }
    if (k < 0) {
        k += arr.shape[2];
    }
    if (l < 0) {
        l += arr.shape[3];
    }

    T& result = *grad_at_byte_offset(arr, byte_offset(arr, i, j, k, l));
    FP_VERIFY_FWD_4(result)

    return result;
}


template <typename T> CUDA_CALLABLE inline T& index(const indexedarray_t<T>& iarr, int i)
{
    assert(iarr.arr.ndim == 1);
    assert(i >= -iarr.shape[0] && i < iarr.shape[0]);

    if (i < 0) {
        i += iarr.shape[0];
    }

    if (iarr.indices[0]) {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }

    T& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T& index(const indexedarray_t<T>& iarr, int i, int j)
{
    assert(iarr.arr.ndim == 2);
    assert(i >= -iarr.shape[0] && i < iarr.shape[0]);
    assert(j >= -iarr.shape[1] && j < iarr.shape[1]);

    if (i < 0) {
        i += iarr.shape[0];
    }
    if (j < 0) {
        j += iarr.shape[1];
    }

    if (iarr.indices[0]) {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }
    if (iarr.indices[1]) {
        j = iarr.indices[1][j];
        assert(j >= 0 && j < iarr.arr.shape[1]);
    }

    T& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i, j));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T& index(const indexedarray_t<T>& iarr, int i, int j, int k)
{
    assert(iarr.arr.ndim == 3);
    assert(i >= -iarr.shape[0] && i < iarr.shape[0]);
    assert(j >= -iarr.shape[1] && j < iarr.shape[1]);
    assert(k >= -iarr.shape[2] && k < iarr.shape[2]);

    if (i < 0) {
        i += iarr.shape[0];
    }
    if (j < 0) {
        j += iarr.shape[1];
    }
    if (k < 0) {
        k += iarr.shape[2];
    }

    if (iarr.indices[0]) {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }
    if (iarr.indices[1]) {
        j = iarr.indices[1][j];
        assert(j >= 0 && j < iarr.arr.shape[1]);
    }
    if (iarr.indices[2]) {
        k = iarr.indices[2][k];
        assert(k >= 0 && k < iarr.arr.shape[2]);
    }

    T& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i, j, k));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T& index(const indexedarray_t<T>& iarr, int i, int j, int k, int l)
{
    assert(iarr.arr.ndim == 4);
    assert(i >= -iarr.shape[0] && i < iarr.shape[0]);
    assert(j >= -iarr.shape[1] && j < iarr.shape[1]);
    assert(k >= -iarr.shape[2] && k < iarr.shape[2]);
    assert(l >= -iarr.shape[3] && l < iarr.shape[3]);

    if (i < 0) {
        i += iarr.shape[0];
    }
    if (j < 0) {
        j += iarr.shape[1];
    }
    if (k < 0) {
        k += iarr.shape[2];
    }
    if (l < 0) {
        l += iarr.shape[3];
    }

    if (iarr.indices[0]) {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }
    if (iarr.indices[1]) {
        j = iarr.indices[1][j];
        assert(j >= 0 && j < iarr.arr.shape[1]);
    }
    if (iarr.indices[2]) {
        k = iarr.indices[2][k];
        assert(k >= 0 && k < iarr.arr.shape[2]);
    }
    if (iarr.indices[3]) {
        l = iarr.indices[3][l];
        assert(l >= 0 && l < iarr.arr.shape[3]);
    }

    T& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i, j, k, l));
    FP_VERIFY_FWD_1(result)

    return result;
}


template <typename T> CUDA_CALLABLE inline array_t<T> view(array_t<T>& src, int i)
{
    assert(src.ndim > 1);
    assert(i >= -src.shape[0] && i < src.shape[0]);

    if (i < 0) {
        i += src.shape[0];
    }

    array_t<T> a;
    size_t offset = byte_offset(src, i);
    a.data = data_at_byte_offset(src, offset);
    if (src.grad)
        a.grad = grad_at_byte_offset(src, offset);
    a.shape[0] = src.shape[1];
    a.shape[1] = src.shape[2];
    a.shape[2] = src.shape[3];
    a.strides[0] = src.strides[1];
    a.strides[1] = src.strides[2];
    a.strides[2] = src.strides[3];
    a.ndim = src.ndim - 1;

    return a;
}

template <typename T> CUDA_CALLABLE inline array_t<T> view(array_t<T>& src, int i, int j)
{
    assert(src.ndim > 2);
    assert(i >= -src.shape[0] && i < src.shape[0]);
    assert(j >= -src.shape[1] && j < src.shape[1]);

    if (i < 0) {
        i += src.shape[0];
    }
    if (j < 0) {
        j += src.shape[1];
    }

    array_t<T> a;
    size_t offset = byte_offset(src, i, j);
    a.data = data_at_byte_offset(src, offset);
    if (src.grad)
        a.grad = grad_at_byte_offset(src, offset);
    a.shape[0] = src.shape[2];
    a.shape[1] = src.shape[3];
    a.strides[0] = src.strides[2];
    a.strides[1] = src.strides[3];
    a.ndim = src.ndim - 2;

    return a;
}

template <typename T> CUDA_CALLABLE inline array_t<T> view(array_t<T>& src, int i, int j, int k)
{
    assert(src.ndim > 3);
    assert(i >= -src.shape[0] && i < src.shape[0]);
    assert(j >= -src.shape[1] && j < src.shape[1]);
    assert(k >= -src.shape[2] && k < src.shape[2]);

    if (i < 0) {
        i += src.shape[0];
    }
    if (j < 0) {
        j += src.shape[1];
    }
    if (k < 0) {
        k += src.shape[2];
    }

    array_t<T> a;
    size_t offset = byte_offset(src, i, j, k);
    a.data = data_at_byte_offset(src, offset);
    if (src.grad)
        a.grad = grad_at_byte_offset(src, offset);
    a.shape[0] = src.shape[3];
    a.strides[0] = src.strides[3];
    a.ndim = src.ndim - 3;

    return a;
}


// view() overloads for baked_array_t with int indices.  The result is
// itself a baked_array_t whose template args are shifted dims/strides
// from the source — the new shape/strides are pure rearrangements of
// the source's compile-time values, so they stay template-encoded.
// Only the runtime data pointer (offset by the index args) carries
// runtime info.  No `array_t<T>` materialisation, no `shape_t` field
// writes in the C++ source.  Downstream consumers taking `array_t<T>&`
// accept the returned baked_array_t via base-class upcast (the
// inherited fields are populated by the returned baked_array_t's
// ctor — those writes are constexpr and NVRTC folds them).
template <typename T, int Ndim, int S0, int S1, int S2, int S3, int St0, int St1, int St2, int St3>
CUDA_CALLABLE inline baked_array_t<T, Ndim - 1, S1, S2, S3, 0, St1, St2, St3, 0>
view(const baked_array_t<T, Ndim, S0, S1, S2, S3, St0, St1, St2, St3>& src, int i)
{
    static_assert(Ndim > 1, "view(arr, int) requires ndim > 1");
    assert(i >= -S0 && i < S0);

    if (i < 0) {
        i += S0;
    }

    return { (T*)((char*)src.data + (size_t)i * St0) };
}

template <typename T, int Ndim, int S0, int S1, int S2, int S3, int St0, int St1, int St2, int St3>
CUDA_CALLABLE inline baked_array_t<T, Ndim - 2, S2, S3, 0, 0, St2, St3, 0, 0>
view(const baked_array_t<T, Ndim, S0, S1, S2, S3, St0, St1, St2, St3>& src, int i, int j)
{
    static_assert(Ndim > 2, "view(arr, int, int) requires ndim > 2");
    assert(i >= -S0 && i < S0);
    assert(j >= -S1 && j < S1);

    if (i < 0) {
        i += S0;
    }
    if (j < 0) {
        j += S1;
    }

    return { (T*)((char*)src.data + (size_t)i * St0 + (size_t)j * St1) };
}

template <typename T, int Ndim, int S0, int S1, int S2, int S3, int St0, int St1, int St2, int St3>
CUDA_CALLABLE inline baked_array_t<T, Ndim - 3, S3, 0, 0, 0, St3, 0, 0, 0>
view(const baked_array_t<T, Ndim, S0, S1, S2, S3, St0, St1, St2, St3>& src, int i, int j, int k)
{
    static_assert(Ndim > 3, "view(arr, int, int, int) requires ndim > 3");
    assert(i >= -S0 && i < S0);
    assert(j >= -S1 && j < S1);
    assert(k >= -S2 && k < S2);

    if (i < 0) {
        i += S0;
    }
    if (j < 0) {
        j += S1;
    }
    if (k < 0) {
        k += S2;
    }

    return { (T*)((char*)src.data + (size_t)i * St0 + (size_t)j * St1 + (size_t)k * St2) };
}


template <typename T, size_t... Idxs>
size_t byte_offset_helper(array_t<T>& src, const slice_t (&slices)[sizeof...(Idxs)], index_sequence<Idxs...>)
{
    return byte_offset(src, slices[Idxs].start...);
}


template <typename T, typename... Slices>
CUDA_CALLABLE inline array_t<T> view(array_t<T>& src, const Slices&... slice_args)
{
    constexpr int N = sizeof...(Slices);
    static_assert(N >= 1 && N <= 4, "view supports 1 to 4 slices");
    assert(src.ndim >= N);

    slice_t slices[N] = { slice_args... };
    int slice_idxs[N];
    int slice_count = 0;

    for (int i = 0; i < N; ++i) {
        if (slices[i].step == 0) {
            // We have a slice representing an integer index.
            if (slices[i].start < 0) {
                slices[i].start += src.shape[i];
            }
        } else {
            slices[i] = slice_adjust_indices(slices[i], src.shape[i]);
            slice_idxs[slice_count] = i;
            ++slice_count;
        }
    }

    size_t offset = byte_offset_helper(src, slices, make_index_sequence<N> {});

    array_t<T> out;

    out.data = data_at_byte_offset(src, offset);
    if (src.grad) {
        out.grad = grad_at_byte_offset(src, offset);
    }

    int dim = 0;
    for (; dim < slice_count; ++dim) {
        int idx = slice_idxs[dim];
        out.shape[dim] = slice_get_length(slices[idx]);
        out.strides[dim] = src.strides[idx] * slices[idx].step;
    }
    for (; dim < slice_count + 4 - N; ++dim) {
        out.shape[dim] = src.shape[dim - slice_count + N];
        out.strides[dim] = src.strides[dim - slice_count + N];
    }
    for (; dim < 4; ++dim) {
        out.shape[dim] = 0;
        out.strides[dim] = 0;
    }

    out.ndim = src.ndim + slice_count - N;
    return out;
}

template <typename T> CUDA_CALLABLE inline indexedarray_t<T> view(indexedarray_t<T>& src, int i)
{
    assert(src.arr.ndim > 1);

    if (src.indices[0]) {
        assert(i >= -src.shape[0] && i < src.shape[0]);
        if (i < 0) {
            i += src.shape[0];
        }
        i = src.indices[0][i];
    }

    indexedarray_t<T> a;
    a.arr = view(src.arr, i);
    a.indices[0] = src.indices[1];
    a.indices[1] = src.indices[2];
    a.indices[2] = src.indices[3];
    a.shape[0] = src.shape[1];
    a.shape[1] = src.shape[2];
    a.shape[2] = src.shape[3];

    return a;
}

template <typename T> CUDA_CALLABLE inline indexedarray_t<T> view(indexedarray_t<T>& src, int i, int j)
{
    assert(src.arr.ndim > 2);

    if (src.indices[0]) {
        assert(i >= -src.shape[0] && i < src.shape[0]);
        if (i < 0) {
            i += src.shape[0];
        }
        i = src.indices[0][i];
    }
    if (src.indices[1]) {
        assert(j >= -src.shape[1] && j < src.shape[1]);
        if (j < 0) {
            j += src.shape[1];
        }
        j = src.indices[1][j];
    }

    indexedarray_t<T> a;
    a.arr = view(src.arr, i, j);
    a.indices[0] = src.indices[2];
    a.indices[1] = src.indices[3];
    a.shape[0] = src.shape[2];
    a.shape[1] = src.shape[3];

    return a;
}

template <typename T> CUDA_CALLABLE inline indexedarray_t<T> view(indexedarray_t<T>& src, int i, int j, int k)
{
    assert(src.arr.ndim > 3);

    if (src.indices[0]) {
        assert(i >= -src.shape[0] && i < src.shape[0]);
        if (i < 0) {
            i += src.shape[0];
        }
        i = src.indices[0][i];
    }
    if (src.indices[1]) {
        assert(j >= -src.shape[1] && j < src.shape[1]);
        if (j < 0) {
            j += src.shape[1];
        }
        j = src.indices[1][j];
    }
    if (src.indices[2]) {
        assert(k >= -src.shape[2] && k < src.shape[2]);
        if (k < 0) {
            k += src.shape[2];
        }
        k = src.indices[2][k];
    }

    indexedarray_t<T> a;
    a.arr = view(src.arr, i, j, k);
    a.indices[0] = src.indices[3];
    a.shape[0] = src.shape[3];

    return a;
}

template <template <typename> class A1, template <typename> class A2, template <typename> class A3, typename T>
inline CUDA_CALLABLE void adj_view(A1<T>& src, int i, A2<T>& adj_src, int adj_i, A3<T>& adj_ret)
{
}
template <template <typename> class A1, template <typename> class A2, template <typename> class A3, typename T>
inline CUDA_CALLABLE void adj_view(A1<T>& src, int i, int j, A2<T>& adj_src, int adj_i, int adj_j, A3<T>& adj_ret)
{
}
template <template <typename> class A1, template <typename> class A2, template <typename> class A3, typename T>
inline CUDA_CALLABLE void
adj_view(A1<T>& src, int i, int j, int k, A2<T>& adj_src, int adj_i, int adj_j, int adj_k, A3<T>& adj_ret)
{
}

template <typename... Args> CUDA_CALLABLE inline void adj_view(Args&&...) { }

// TODO: lower_bound() for indexed arrays?

template <typename T> CUDA_CALLABLE inline int lower_bound(const array_t<T>& arr, int arr_begin, int arr_end, T value)
{
    assert(arr.ndim == 1);

    int lower = arr_begin;
    int upper = arr_end - 1;

    while (lower < upper) {
        int mid = lower + (upper - lower) / 2;

        if (arr[mid] < value) {
            lower = mid + 1;
        } else {
            upper = mid;
        }
    }

    return lower;
}

template <typename T> CUDA_CALLABLE inline int lower_bound(const array_t<T>& arr, T value)
{
    return lower_bound(arr, 0, arr.shape[0], value);
}

template <typename T>
inline CUDA_CALLABLE void adj_lower_bound(const array_t<T>& arr, T value, array_t<T> adj_arr, T adj_value, int adj_ret)
{
}
template <typename T>
inline CUDA_CALLABLE void adj_lower_bound(
    const array_t<T>& arr,
    int arr_begin,
    int arr_end,
    T value,
    array_t<T> adj_arr,
    int adj_arr_begin,
    int adj_arr_end,
    T adj_value,
    int adj_ret
)
{
}

template <template <typename> class A, typename T> inline CUDA_CALLABLE T atomic_add(const A<T>& buf, int i, T value)
{
    return atomic_add(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_add(const A<T>& buf, int i, int j, T value)
{
    return atomic_add(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_add(const A<T>& buf, int i, int j, int k, T value)
{
    return atomic_add(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_add(const A<T>& buf, int i, int j, int k, int l, T value)
{
    return atomic_add(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T> inline CUDA_CALLABLE T atomic_sub(const A<T>& buf, int i, T value)
{
    return atomic_add(&index(buf, i), -value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_sub(const A<T>& buf, int i, int j, T value)
{
    return atomic_add(&index(buf, i, j), -value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_sub(const A<T>& buf, int i, int j, int k, T value)
{
    return atomic_add(&index(buf, i, j, k), -value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_sub(const A<T>& buf, int i, int j, int k, int l, T value)
{
    return atomic_add(&index(buf, i, j, k, l), -value);
}

template <template <typename> class A, typename T> inline CUDA_CALLABLE T atomic_min(const A<T>& buf, int i, T value)
{
    return atomic_min(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_min(const A<T>& buf, int i, int j, T value)
{
    return atomic_min(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_min(const A<T>& buf, int i, int j, int k, T value)
{
    return atomic_min(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_min(const A<T>& buf, int i, int j, int k, int l, T value)
{
    return atomic_min(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T> inline CUDA_CALLABLE T atomic_max(const A<T>& buf, int i, T value)
{
    return atomic_max(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_max(const A<T>& buf, int i, int j, T value)
{
    return atomic_max(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_max(const A<T>& buf, int i, int j, int k, T value)
{
    return atomic_max(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_max(const A<T>& buf, int i, int j, int k, int l, T value)
{
    return atomic_max(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_cas(const A<T>& buf, int i, T old_value, T new_value)
{
    return atomic_cas(&index(buf, i), old_value, new_value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_cas(const A<T>& buf, int i, int j, T old_value, T new_value)
{
    return atomic_cas(&index(buf, i, j), old_value, new_value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_cas(const A<T>& buf, int i, int j, int k, T old_value, T new_value)
{
    return atomic_cas(&index(buf, i, j, k), old_value, new_value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_cas(const A<T>& buf, int i, int j, int k, int l, T old_value, T new_value)
{
    return atomic_cas(&index(buf, i, j, k, l), old_value, new_value);
}

template <template <typename> class A, typename T> inline CUDA_CALLABLE T atomic_exch(const A<T>& buf, int i, T value)
{
    return atomic_exch(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_exch(const A<T>& buf, int i, int j, T value)
{
    return atomic_exch(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_exch(const A<T>& buf, int i, int j, int k, T value)
{
    return atomic_exch(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_exch(const A<T>& buf, int i, int j, int k, int l, T value)
{
    return atomic_exch(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T> inline CUDA_CALLABLE T atomic_and(const A<T>& buf, int i, T value)
{
    return atomic_and(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_and(const A<T>& buf, int i, int j, T value)
{
    return atomic_and(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_and(const A<T>& buf, int i, int j, int k, T value)
{
    return atomic_and(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_and(const A<T>& buf, int i, int j, int k, int l, T value)
{
    return atomic_and(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T> inline CUDA_CALLABLE T atomic_or(const A<T>& buf, int i, T value)
{
    return atomic_or(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_or(const A<T>& buf, int i, int j, T value)
{
    return atomic_or(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_or(const A<T>& buf, int i, int j, int k, T value)
{
    return atomic_or(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_or(const A<T>& buf, int i, int j, int k, int l, T value)
{
    return atomic_or(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T> inline CUDA_CALLABLE T atomic_xor(const A<T>& buf, int i, T value)
{
    return atomic_xor(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_xor(const A<T>& buf, int i, int j, T value)
{
    return atomic_xor(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_xor(const A<T>& buf, int i, int j, int k, T value)
{
    return atomic_xor(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_xor(const A<T>& buf, int i, int j, int k, int l, T value)
{
    return atomic_xor(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T> inline CUDA_CALLABLE T* address(const A<T>& buf, int i)
{
    return &index(buf, i);  // cppcheck-suppress returnDanglingLifetime
}
template <template <typename> class A, typename T> inline CUDA_CALLABLE T* address(const A<T>& buf, int i, int j)
{
    return &index(buf, i, j);  // cppcheck-suppress returnDanglingLifetime
}
template <template <typename> class A, typename T> inline CUDA_CALLABLE T* address(const A<T>& buf, int i, int j, int k)
{
    return &index(buf, i, j, k);  // cppcheck-suppress returnDanglingLifetime
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T* address(const A<T>& buf, int i, int j, int k, int l)
{
    return &index(buf, i, j, k, l);  // cppcheck-suppress returnDanglingLifetime
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T>& buf, int i, T value)
{
    FP_VERIFY_FWD_1(value)

    index(buf, i) = value;
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T>& buf, int i, int j, T value)
{
    FP_VERIFY_FWD_2(value)

    index(buf, i, j) = value;
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T>& buf, int i, int j, int k, T value)
{
    FP_VERIFY_FWD_3(value)

    index(buf, i, j, k) = value;
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T>& buf, int i, int j, int k, int l, T value)
{
    FP_VERIFY_FWD_4(value)

    index(buf, i, j, k, l) = value;
}

template <typename T> inline CUDA_CALLABLE void store(T* address, T value)
{
    FP_VERIFY_FWD(value)

    *address = value;
}

// Specific overload for storing a baked_array_t value into an
// `array_t<T>*` slot (e.g. `geom1.vert = mesh_vert` where the
// struct field is `array_t<vec3>` and the kernel arg `mesh_vert` is
// declared as `baked_array_t<vec3, ...>` after Phase J).  Without
// this, `wp::store(T*, T)` fails template deduction because the two
// args' Ts (`array_t<vec3>` vs `baked_array_t<vec3, ...>`) don't
// match — the implicit-conversion path can't be used during
// deduction.  The slicing assignment to the base subobject is what
// the array_t<T> case did before Phase J.
template <typename T, int Ndim, int S0, int S1, int S2, int S3, int St0, int St1, int St2, int St3>
inline CUDA_CALLABLE void
store(array_t<T>* address, const baked_array_t<T, Ndim, S0, S1, S2, S3, St0, St1, St2, St3>& value)
{
    *address = value;
}

template <typename T> inline CUDA_CALLABLE T load(T* address)
{
    T value = *address;
    FP_VERIFY_FWD(value)

    return value;
}

// where() overload for array condition - returns a if array.data is non-null, otherwise returns b
template <typename T1, typename T2> CUDA_CALLABLE inline T2 where(const array_t<T1>& arr, const T2& a, const T2& b)
{
    return arr.data ? a : b;
}

template <typename T1, typename T2>
CUDA_CALLABLE inline void adj_where(
    const array_t<T1>& arr,
    const T2& a,
    const T2& b,
    const array_t<T1>& adj_cond,
    T2& adj_a,
    T2& adj_b,
    const T2& adj_ret
)
{
    if (arr.data)
        adj_a += adj_ret;
    else
        adj_b += adj_ret;
}

// stub for the case where we have an nested array inside a struct and
// atomic add the whole struct onto an array (e.g.: during backwards pass)
template <typename T> CUDA_CALLABLE inline void atomic_add(array_t<T>*, array_t<T>) { }

// stub for the case where we have an indexed array inside a struct and
// atomic add the whole struct onto an array (e.g.: during backwards pass)
template <typename T> CUDA_CALLABLE inline void atomic_add(indexedarray_t<T>*, indexedarray_t<T>) { }

// for float and vector types this is just an alias for an atomic add
template <typename T> CUDA_CALLABLE inline void adj_atomic_add(T* buf, T value) { atomic_add(buf, value); }


// for integral types we do not accumulate gradients
CUDA_CALLABLE inline void adj_atomic_add(int8* buf, int8 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint8* buf, uint8 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int16* buf, int16 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint16* buf, uint16 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int32* buf, int32 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint32* buf, uint32 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int64* buf, int64 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint64* buf, uint64 value) { }

CUDA_CALLABLE inline void adj_atomic_add(bool* buf, bool value) { }

// only generate gradients for T types
template <typename T>
inline CUDA_CALLABLE void
adj_address(const array_t<T>& buf, int i, const array_t<T>& adj_buf, int adj_i, const T& adj_output)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i), adj_output);
}
template <typename T>
inline CUDA_CALLABLE void
adj_address(const array_t<T>& buf, int i, int j, const array_t<T>& adj_buf, int adj_i, int adj_j, const T& adj_output)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i, j), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i, j), adj_output);
}
template <typename T>
inline CUDA_CALLABLE void adj_address(
    const array_t<T>& buf,
    int i,
    int j,
    int k,
    const array_t<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    const T& adj_output
)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i, j, k), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i, j, k), adj_output);
}
template <typename T>
inline CUDA_CALLABLE void adj_address(
    const array_t<T>& buf,
    int i,
    int j,
    int k,
    int l,
    const array_t<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    const T& adj_output
)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i, j, k, l), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i, j, k, l), adj_output);
}

template <typename T>
inline CUDA_CALLABLE void
adj_array_store(const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int adj_i, T& adj_value)
{
    if (adj_buf.data) {
        T& g = index(adj_buf, i);
        adj_value += g;

        // Only zero if adj_buf aliases buf.grad (standard Warp Tape case)
        // and retain_grad is not set on the forward array.
        // Skip zeroing for external gradient buffers passed via adj_inputs,
        // since the caller owns them and may need to preserve accumulated gradients.
        if (buf.grad && adj_buf.data == buf.grad && !(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    } else if (buf.grad) {
        // No explicit adjoint passed (adj_buf is null), fall back to buf.grad.
        T& g = index_grad(buf, i);
        adj_value += g;
        if (!(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    }

    FP_VERIFY_ADJ_1(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_array_store(
    const array_t<T>& buf, int i, int j, T value, const array_t<T>& adj_buf, int adj_i, int adj_j, T& adj_value
)
{
    if (adj_buf.data) {
        T& g = index(adj_buf, i, j);
        adj_value += g;
        if (buf.grad && adj_buf.data == buf.grad && !(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    } else if (buf.grad) {
        T& g = index_grad(buf, i, j);
        adj_value += g;
        if (!(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    }

    FP_VERIFY_ADJ_2(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_array_store(
    const array_t<T>& buf,
    int i,
    int j,
    int k,
    T value,
    const array_t<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_value
)
{
    if (adj_buf.data) {
        T& g = index(adj_buf, i, j, k);
        adj_value += g;
        if (buf.grad && adj_buf.data == buf.grad && !(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    } else if (buf.grad) {
        T& g = index_grad(buf, i, j, k);
        adj_value += g;
        if (!(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    }

    FP_VERIFY_ADJ_3(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_array_store(
    const array_t<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const array_t<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value
)
{
    if (adj_buf.data) {
        T& g = index(adj_buf, i, j, k, l);
        adj_value += g;
        if (buf.grad && adj_buf.data == buf.grad && !(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    } else if (buf.grad) {
        T& g = index_grad(buf, i, j, k, l);
        adj_value += g;
        if (!(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    }

    FP_VERIFY_ADJ_4(value, adj_value)
}

template <typename T> inline CUDA_CALLABLE void adj_store(const T* address, T value, const T& adj_address, T& adj_value)
{
    // nop; generic store() operations are not differentiable, only array_store() is
    FP_VERIFY_ADJ(value, adj_value)
}

template <typename T> inline CUDA_CALLABLE void adj_load(const T* address, const T& adj_address, T& adj_value)
{
    // nop; generic load() operations are not differentiable
}

template <typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret
)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i);
    else if (buf.grad)
        adj_value += index_grad(buf, i);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const array_t<T>& buf,
    int i,
    int j,
    T value,
    const array_t<T>& adj_buf,
    int adj_i,
    int adj_j,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j);
    else if (buf.grad)
        adj_value += index_grad(buf, i, j);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const array_t<T>& buf,
    int i,
    int j,
    int k,
    T value,
    const array_t<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k);
    else if (buf.grad)
        adj_value += index_grad(buf, i, j, k);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const array_t<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const array_t<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_value += index(adj_buf, i, j, k, l);
    else if (buf.grad)
        adj_value += index_grad(buf, i, j, k, l);

    FP_VERIFY_ADJ_4(value, adj_value)
}

template <typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const array_t<T>& buf, int i, T value, const array_t<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret
)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i);
    else if (buf.grad)
        adj_value -= index_grad(buf, i);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const array_t<T>& buf,
    int i,
    int j,
    T value,
    const array_t<T>& adj_buf,
    int adj_i,
    int adj_j,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j);
    else if (buf.grad)
        adj_value -= index_grad(buf, i, j);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const array_t<T>& buf,
    int i,
    int j,
    int k,
    T value,
    const array_t<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j, k);
    else if (buf.grad)
        adj_value -= index_grad(buf, i, j, k);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const array_t<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const array_t<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_value -= index(adj_buf, i, j, k, l);
    else if (buf.grad)
        adj_value -= index_grad(buf, i, j, k, l);

    FP_VERIFY_ADJ_4(value, adj_value)
}

// generic array types that do not support gradient computation (indexedarray, etc.)
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_address(const A1<T>& buf, int i, const A2<T>& adj_buf, int adj_i, const T& adj_output)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_address(const A1<T>& buf, int i, int j, const A2<T>& adj_buf, int adj_i, int adj_j, const T& adj_output)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_address(
    const A1<T>& buf, int i, int j, int k, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, const T& adj_output
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_address(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    int l,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    const T& adj_output
)
{
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_array_store(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_array_store(const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_array_store(
    const A1<T>& buf, int i, int j, int k, T value, const A2<T>& adj_buf, int adj_i, int adj_j, int adj_k, T& adj_value
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_array_store(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value
)
{
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_atomic_add(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_value,
    const T& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value,
    const T& adj_ret
)
{
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_atomic_sub(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_value,
    const T& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value,
    const T& adj_ret
)
{
}

// generic handler for scalar values
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_atomic_min(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i), &index(adj_buf, i), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i), &index_grad(buf, i), value, adj_value);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_min(
    const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j), &index(adj_buf, i, j), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j), &index_grad(buf, i, j), value, adj_value);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_min(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k), &index(adj_buf, i, j, k), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k), &index_grad(buf, i, j, k), value, adj_value);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_min(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index(adj_buf, i, j, k, l), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index_grad(buf, i, j, k, l), value, adj_value);

    FP_VERIFY_ADJ_4(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_atomic_max(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i), &index(adj_buf, i), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i), &index_grad(buf, i), value, adj_value);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_max(
    const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j), &index(adj_buf, i, j), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j), &index_grad(buf, i, j), value, adj_value);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_max(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k), &index(adj_buf, i, j, k), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k), &index_grad(buf, i, j, k), value, adj_value);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_max(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index(adj_buf, i, j, k, l), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index_grad(buf, i, j, k, l), value, adj_value);

    FP_VERIFY_ADJ_4(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_cas(
    const A1<T>& buf,
    int i,
    T compare,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    T& adj_compare,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_cas(&index(buf, i), compare, value, &index(adj_buf, i), adj_compare, adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_cas(&index(buf, i), compare, value, &index_grad(buf, i), adj_compare, adj_value, adj_ret);

    FP_VERIFY_ADJ_1(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_cas(
    const A1<T>& buf,
    int i,
    int j,
    T compare,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    T& adj_compare,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_cas(&index(buf, i, j), compare, value, &index(adj_buf, i, j), adj_compare, adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_cas(&index(buf, i, j), compare, value, &index_grad(buf, i, j), adj_compare, adj_value, adj_ret);

    FP_VERIFY_ADJ_2(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_cas(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    T compare,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_compare,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_cas(&index(buf, i, j, k), compare, value, &index(adj_buf, i, j, k), adj_compare, adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_cas(
            &index(buf, i, j, k), compare, value, &index_grad(buf, i, j, k), adj_compare, adj_value, adj_ret
        );

    FP_VERIFY_ADJ_3(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_cas(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T compare,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_compare,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_cas(
            &index(buf, i, j, k, l), compare, value, &index(adj_buf, i, j, k, l), adj_compare, adj_value, adj_ret
        );
    else if (buf.grad)
        adj_atomic_cas(
            &index(buf, i, j, k, l), compare, value, &index_grad(buf, i, j, k, l), adj_compare, adj_value, adj_ret
        );

    FP_VERIFY_ADJ_4(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_atomic_exch(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret)
{
    if (adj_buf.data)
        adj_atomic_exch(&index(buf, i), value, &index(adj_buf, i), adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_exch(&index(buf, i), value, &index_grad(buf, i), adj_value, adj_ret);

    FP_VERIFY_ADJ_1(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_exch(
    const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_exch(&index(buf, i, j), value, &index(adj_buf, i, j), adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_exch(&index(buf, i, j), value, &index_grad(buf, i, j), adj_value, adj_ret);

    FP_VERIFY_ADJ_2(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_exch(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_exch(&index(buf, i, j, k), value, &index(adj_buf, i, j, k), adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_exch(&index(buf, i, j, k), value, &index_grad(buf, i, j, k), adj_value, adj_ret);

    FP_VERIFY_ADJ_3(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_exch(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value,
    const T& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_exch(&index(buf, i, j, k, l), value, &index(adj_buf, i, j, k, l), adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_exch(&index(buf, i, j, k, l), value, &index_grad(buf, i, j, k, l), adj_value, adj_ret);

    FP_VERIFY_ADJ_4(value, adj_value)
}

// for bitwise operations we do not accumulate gradients
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_atomic_and(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_and(
    const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_and(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_value,
    const T& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_and(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value,
    const T& adj_ret
)
{
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_atomic_or(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_or(
    const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_or(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_value,
    const T& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_or(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value,
    const T& adj_ret
)
{
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_atomic_xor(const A1<T>& buf, int i, T value, const A2<T>& adj_buf, int adj_i, T& adj_value, const T& adj_ret)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_xor(
    const A1<T>& buf, int i, int j, T value, const A2<T>& adj_buf, int adj_i, int adj_j, T& adj_value, const T& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_xor(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T& adj_value,
    const T& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_xor(
    const A1<T>& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T>& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T& adj_value,
    const T& adj_ret
)
{
}


template <template <typename> class A, typename T> CUDA_CALLABLE inline int len(const A<T>& a) { return a.shape[0]; }

template <template <typename> class A, typename T>
CUDA_CALLABLE inline void adj_len(const A<T>& a, A<T>& adj_a, int& adj_ret)
{
}

}  // namespace wp

#include "fabric.h"
