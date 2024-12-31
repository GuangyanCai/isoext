#pragma once

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <thrust/device_vector.h>

#include "math.cuh"

// Function to copy data from device to host
template <typename T>
T *
device_to_host(const T *d_ptr, size_t size) {
    T *h_ptr = (T *) malloc(size * sizeof(T));
    cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    return h_ptr;
}

// Function to copy data from host to device
template <typename T>
T *
host_to_device(const T *h_ptr, size_t size) {
    T *d_ptr;
    cudaMalloc((void **) &d_ptr, size * sizeof(T));
    cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    return d_ptr;
}

struct idx_to_cell_op {
    uint *cells;
    uint3 shape;

    __host__ __device__ idx_to_cell_op(uint *cells, uint3 shape)
        : cells(cells), shape(shape) {}

    __host__ __device__ void operator()(uint idx) {
        uint x, y, z, yz, i;
        i = idx;
        z = i % (shape.z - 1);
        i /= (shape.z - 1);
        y = i % (shape.y - 1);
        x = i / (shape.y - 1);
        idx *= 8;
        yz = shape.y * shape.z;
        cells[idx + 0] = x * yz + y * shape.z + z;
        cells[idx + 1] = cells[idx + 0] + 1;
        cells[idx + 2] = cells[idx + 0] + shape.z;
        cells[idx + 3] = cells[idx + 1] + shape.z;
        cells[idx + 4] = cells[idx + 0] + yz;
        cells[idx + 5] = cells[idx + 1] + yz;
        cells[idx + 6] = cells[idx + 2] + yz;
        cells[idx + 7] = cells[idx + 3] + yz;
    }
};

struct get_vtx_pos_op {
    const uint3 res;
    const float3 aabb_min;
    const float3 aabb_size;

    get_vtx_pos_op(const uint3 res, const float3 aabb_min,
                   const float3 aabb_max)
        : res(res), aabb_min(aabb_min), aabb_size(aabb_max - aabb_min) {}

    __host__ __device__ float3 operator()(uint idx) {
        float3 pos;
        pos.z = (idx % res.z) / (float) (res.z - 1);
        idx /= res.z;
        pos.y = (idx % res.y) / (float) (res.y - 1);
        pos.x = (idx / res.y) / (float) (res.x - 1);
        pos = aabb_min + pos * aabb_size;
        return pos;
    }
};

struct get_case_idx_op {
    uint8_t *cases;
    const float *values;
    const uint *cells;
    const float level;

    get_case_idx_op(uint8_t *cases, const float *values, const uint *cells,
                    const float level)
        : cases(cases), values(values), cells(cells), level(level) {}

    __host__ __device__ void operator()(uint32_t cell_idx) {
        // Compute the sign of each cube vertex and derive the case number
        uint8_t case_num = 0;
        uint offset = cell_idx * 8;
        for (uint i = 0; i < 8; i++) {
            float p_val = values[cells[offset + i]];
            case_num |= (p_val - level < 0) << i;
        }
        cases[cell_idx] = case_num;
    }
};

void vertex_welding(thrust::device_vector<float3> &v,
                    thrust::device_vector<int> &f, bool skip_scatter = true);