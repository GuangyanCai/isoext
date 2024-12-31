#pragma once

#include <thrust/copy.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <array>
#include <numeric>
#include <vector>

template <typename DTYPE> struct NDArray {
    thrust::device_ptr<DTYPE> data_ptr;
    std::vector<size_t> shape;
    bool read_only;

    NDArray(thrust::device_ptr<DTYPE> data_ptr, std::vector<size_t> shape,
            bool read_only = true)
        : data_ptr(data_ptr), shape(shape), read_only(read_only) {}

    NDArray(DTYPE *data, std::vector<size_t> shape, bool read_only = true)
        : data_ptr(data), shape(shape), read_only(read_only) {}

    NDArray(std::vector<size_t> shape) : shape(shape), read_only(false) {
        data_ptr = thrust::device_malloc<DTYPE>(size());
    }

    NDArray(const NDArray<DTYPE> &other)
        : shape(other.shape), read_only(false) {
        data_ptr = thrust::device_malloc<DTYPE>(other.size());
        thrust::copy(other.data_ptr, other.data_ptr + other.size(), data_ptr);
    }

    NDArray(NDArray<DTYPE> &&other) noexcept
        : data_ptr(other.data_ptr), shape(std::move(other.shape)),
          read_only(other.read_only) {
        other.data_ptr = nullptr;
        other.read_only = true;
    }

    ~NDArray() {
        if (!read_only) {
            thrust::device_free(data_ptr);
        }
    }

    inline size_t size() const {
        return std::accumulate(shape.begin(), shape.end(), size_t{1},
                               std::multiplies<size_t>());
    }

    inline size_t ndim() const { return shape.size(); }

    DTYPE *data() const { return data_ptr.get(); }

    void set(const NDArray<DTYPE> &other) {
        if (shape != other.shape) {
            throw std::runtime_error("Cannot set values with different shapes");
        }
        thrust::copy(other.data_ptr, other.data_ptr + size(), data_ptr);
    }

    void set(const thrust::device_vector<DTYPE> dv) {
        if (size() != dv.size()) {
            throw std::runtime_error("Cannot set values with different sizes");
        }
        thrust::copy(dv.begin(), dv.end(), data_ptr);
    }

    static NDArray<DTYPE> copy(const DTYPE *data, std::vector<size_t> shape) {
        NDArray<DTYPE> arr(shape);
        thrust::copy(data, data + arr.size(), arr.data_ptr);
        return arr;
    }

    void fill(DTYPE value) { thrust::fill(data_ptr, data_ptr + size(), value); }

    template <typename NEW_DTYPE> NDArray<NEW_DTYPE> cast() {
        NDArray<NEW_DTYPE> arr(shape);
        thrust::transform(
            data_ptr, data_ptr + size(), arr.data_ptr,
            [] __device__(DTYPE x) { return static_cast<NEW_DTYPE>(x); });
        return arr;
    }
};

// Our cell is defined as follows:
//        v3------e10-----v7
//       /|               /|
//      / |              / |
//    e1  |            e5  |
//    /  e2            /   e6
//   /    |           /    |
//  v1------e9------v5     |
//  |     |          |     |
//  |    v2------e11-|----v6
//  |    /           |    /
// e0  e3           e4  e7
//  |  /             |  /
//  | /              | /
//  |/               |/
//  v0------e8------v4
//
//  z
//  |  y
//  | /
//  |/
//  +----x
//
// This ASCII art represents a 3D cube with:
// - Vertices labeled v0 to v7 in Morton order
// - Edges labeled e0 to e11
// - Front, top, and right faces visible
//
// Vertex mapping in Morton order:
// v0: (0,0,0)  v1: (0,0,1)  v2: (0,1,0)  v3: (0,1,1)
// v4: (1,0,0)  v5: (1,0,1)  v6: (1,1,0)  v7: (1,1,1)
//
// Edge mapping:
// e0: v0-v1   e1: v1-v3   e2: v2-v3   e3: v0-v2
// e4: v4-v5   e5: v5-v7   e6: v6-v7   e7: v4-v6
// e8: v0-v4   e9: v1-v5   e10: v3-v7  e11: v2-v6

// LUTs for cells.

// LUT for edges. Every two elements define an edge.
constexpr size_t edges_size = 24;
extern const int edges[edges_size];

// LUT for edge intersection status.
constexpr size_t edge_table_size = 256;
extern const int edge_table[edge_table_size];
