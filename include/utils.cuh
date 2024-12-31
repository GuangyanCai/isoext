#pragma once

#include "common.cuh"
#include "grid/grid.cuh"
#include "math.cuh"

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <thrust/device_vector.h>

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

struct get_edge_status_op {
    int *edge_status;
    const float *values;
    const uint *cells;
    const int *edge_table;
    const float level;

    get_edge_status_op(int *edge_status, const float *values, const uint *cells,
                       const int *edge_table, const float level)
        : edge_status(edge_status), values(values), cells(cells),
          edge_table(edge_table), level(level) {}

    __host__ __device__ void operator()(uint cell_idx) {
        // Compute the sign of each cube vertex and derive the case number
        uint8_t case_num = 0;
        uint offset = cell_idx * 8;
        for (uint i = 0; i < 8; i++) {
            float p_val = values[cells[offset + i]];
            case_num |= (p_val - level < 0) << i;
        }
        edge_status[cell_idx] = edge_table[case_num];
    }
};

struct get_its_points_op {
    float3 *its_points;
    const uint *cell_idx;
    const uint *cell_offsets;
    const int *edge_status;
    const float *values;
    const float3 *points;
    const uint *cells;
    const int *edges;
    const float level;

    get_its_points_op(float3 *its_points, const uint *cell_idx,
                      const uint *cell_offsets, const int *edge_status,
                      const float *values, const float3 *points,
                      const uint *cells, const int *edges, const float level)
        : its_points(its_points), cell_idx(cell_idx),
          cell_offsets(cell_offsets), edge_status(edge_status), values(values),
          points(points), cells(cells), edges(edges), level(level) {}

    __host__ __device__ void operator()(uint idx) {
        int status = edge_status[idx];
        int offset = cell_offsets[idx];

        // Compute the location of each cube vertex.
        float3 c_p[8];
        float c_v[8];
        uint c_offset = cell_idx[idx] * 8;
        for (uint32_t i = 0; i < 8; i++) {
            c_p[i] = points[cells[c_offset + i]];
            c_v[i] = values[cells[c_offset + i]];
        }

        for (int i = 0; i < 12; i++) {
            if (status & (1 << i)) {
                int p_0 = edges[i * 2];
                int p_1 = edges[i * 2 + 1];
                float denom = c_v[p_1] - c_v[p_0];
                float t = (denom != 0.0f) ? (level - c_v[p_0]) / denom : 0.0f;
                its_points[offset] = lerp(t, c_p[p_0], c_p[p_1]);
                offset++;
            }
        }
    }
};

struct Intersection {
    NDArray<float3> points;
    NDArray<float3> normals;
    NDArray<uint> cell_offsets;

    Intersection(uint num_points, NDArray<uint> &&cell_offsets)
        : points({num_points}), normals({num_points}),
          cell_offsets(std::move(cell_offsets)) {}

    inline NDArray<float3> get_points() { return points; }
    inline NDArray<float3> get_normals() { return normals; }
    inline void set_normals(const NDArray<float3> &normals_) {
        normals.set(normals_);
    }
};

void vertex_welding(thrust::device_vector<float3> &v,
                    thrust::device_vector<int> &f, bool skip_scatter = true);

Intersection get_intersection(Grid *grid, float level);