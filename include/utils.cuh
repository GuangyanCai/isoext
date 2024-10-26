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

// Structure representing a cube in a 3D grid
struct Cube {
    uint32_t vi[8];   // indices of the cube vertices in the grid
    uint3 ci;         // 3D cube index

    // Constructor to initialize the cube based on its index and grid resolution
    __host__ __device__ Cube(uint32_t cube_idx, uint3 res, bool tight) {
        // Cube layout:
        //
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

        // When the grid is tight, neighbor cubes share faces. In this case,
        // there are (res.x - 1) * (res.y - 1) * (res.z - 1) cubes.
        if (tight) {
            ci.z = cube_idx % (res.z - 1);
            cube_idx /= (res.z - 1);
            ci.y = cube_idx % (res.y - 1);
            ci.x = cube_idx / (res.y - 1);
        }
        // Otherwise, each cube is separate from others. In this case, res must
        // be (2n, 2, 2), where n is the number of cubes.
        else {
            ci.x = 2 * cube_idx;
            ci.y = 0;
            ci.z = 0;
        }

        // Compute the indices of the cube's vertices in the array.
        uint32_t res_yz = res.y * res.z;
        vi[0] = ci.x * res.y * res.z + ci.y * res.z + ci.z;   // (x, y, z)
        vi[1] = vi[0] + 1;                                    // (x, y, z+1)
        vi[2] = vi[0] + res.z;                                // (x, y+1, z)
        vi[3] = vi[1] + res.z;                                // (x, y+1, z+1)
        vi[4] = vi[0] + res_yz;                               // (x+1, y, z)
        vi[5] = vi[1] + res_yz;                               // (x+1, y, z+1)
        vi[6] = vi[2] + res_yz;                               // (x+1, y+1, z)
        vi[7] = vi[3] + res_yz;                               // (x+1, y+1, z+1)
    }
};

struct get_vtx_pos_op {
    const uint3 res;
    const float3 aabb_min;
    const float3 aabb_size;

    get_vtx_pos_op(const uint3 res, const float3 aabb_min,
                   const float3 aabb_max)
        : res(res), aabb_min(aabb_min), aabb_size(aabb_max - aabb_min) {}

    __host__ __device__ float3 operator()(uint32_t idx) {
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
    const float *grid;
    const uint3 res;
    const float level;
    const bool tight;

    get_case_idx_op(uint8_t *cases, const float *grid, const uint3 res,
                    const float level, const bool tight)
        : cases(cases), grid(grid), res(res), level(level), tight(tight) {}

    __host__ __device__ void operator()(uint32_t cube_idx) {
        // For each cube vertex, compute the index to the grid array.
        Cube c(cube_idx, res, tight);

        // Compute the sign of each cube vertex and derive the case number
        // (table_idx).
        uint8_t table_idx = 0;
        float p_val[8];
        for (uint32_t i = 0; i < 8; i++) {
            p_val[i] = grid[c.vi[i]];
            table_idx |= (p_val[i] - level < 0) << i;
        }
        cases[cube_idx] = table_idx;
    }
};

void vertex_welding(thrust::device_vector<float3> &v,
                    thrust::device_vector<int> &f, bool skip_scatter = true);
