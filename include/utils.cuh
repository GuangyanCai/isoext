#pragma once

#include "math.cuh"

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

__host__ __device__ uint3 idx_1d_to_3d(uint idx, uint3 shape);

__device__ __host__ uint idx_3d_to_1d(uint3 idx, uint3 shape);

__device__ __host__ uint point_idx_to_cell_idx(uint idx, uint3 shape);

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

struct get_case_num_op {
    uint8_t *cases;
    const float *values;
    const uint *cells;
    const float level;

    get_case_num_op(uint8_t *cases, const float *values, const uint *cells,
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

struct edge_to_neighbor_idx_op {
    const uint3 *en_table;
    const uint3 grid_shape;
    const uint3 grid_offset;

    edge_to_neighbor_idx_op(const uint3 *en_table, const uint3 grid_shape)
        : en_table(en_table), grid_shape(grid_shape),
          grid_offset(
              make_uint3(grid_shape.x * grid_shape.y, grid_shape.y, 1)) {}

    __host__ __device__ int4 operator()(uint2 edge) {
        uint3 cell_idx = idx_1d_to_3d(edge.x, grid_shape);

        uint offset = edge.y - edge.x;
        if (offset == grid_offset.x) {
            offset = 0;
        } else if (offset == grid_offset.y) {
            offset = 1;
        } else {
            offset = 2;
        }
        offset *= 4;

        int r[4];
        for (int i = 0; i < 4; i++) {
            uint3 o = en_table[offset + i];
            if (cell_idx.x < o.x || cell_idx.y < o.y || cell_idx.z < o.z) {
                r[i] = -1;
            } else {
                r[i] = idx_3d_to_1d(cell_idx - o, grid_shape - 1);
            }
        }

        return make_int4(r[0], r[1], r[2], r[3]);
    }
};

struct get_its_point_avg_op {
    const float3 *its_points;
    const uint *cell_offsets;

    get_its_point_avg_op(const float3 *its_points, const uint *cell_offsets)
        : its_points(its_points), cell_offsets(cell_offsets) {}

    __host__ __device__ float3 operator()(uint idx) {
        float3 avg = make_float3(0.0f, 0.0f, 0.0f);
        for (uint i = cell_offsets[idx]; i < cell_offsets[idx + 1]; i++) {
            avg = avg + its_points[i];
        }
        avg = avg / (cell_offsets[idx + 1] - cell_offsets[idx]);
        return avg;
    }
};

void vertex_welding(thrust::device_vector<float3> &v,
                    thrust::device_vector<int> &f, bool skip_scatter = true);

thrust::device_vector<int4>
get_edge_neighbors(const thrust::device_vector<uint2> &edges_dv,
                   uint3 grid_shape);