#pragma once

#include <thrust/copy.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>

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

    void set(const NDArray<DTYPE> &other) {
        if (shape != other.shape) {
            throw std::runtime_error("Cannot set values with different shapes");
        }
        thrust::copy(other.data_ptr, other.data_ptr + size(), data_ptr);
    }
};