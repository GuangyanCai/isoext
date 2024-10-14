#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "lorensen.cuh"
#include "lut.cuh"
#include "math.cuh"
#include "utils.cuh"

// Hide helper structures using an anonymous namespace
namespace mc::lorensen {

struct process_cube_op {
    float3 *v;
    const float *grid;
    const int *edges;
    const int *edge_table;
    const int *tri_table;
    const uint3 res;
    const float level;
    const bool tight;

    process_cube_op(float3 *v, const float *grid, const int *edges,
                    const int *edge_table, const int *tri_table,
                    const uint3 res, const float level, const bool tight)
        : v(v), grid(grid), edges(edges), edge_table(edge_table),
          tri_table(tri_table), res(res), level(level), tight(tight) {}

    __host__ __device__ void
    operator()(thrust::tuple<uint8_t, uint32_t, uint32_t> args) {
        uint32_t case_idx, cube_idx, result_idx;
        thrust::tie(case_idx, cube_idx, result_idx) = args;

        // For each cube vertex, compute the index to the grid array.
        Cube c(cube_idx, res, tight);

        // Compute the location of each cube vertex. Assume each cube is a
        // unit cube for now.
        float3 p_pos[8];
        c.get_vtx_pos(p_pos);

        float p_val[8];
        for (uint32_t i = 0; i < 8; i++) {
            p_val[i] = grid[c.vi[i]];
        }

        // Compute the intersection between the isosurface and each edge of
        // the cube.
        int edge_status = edge_table[case_idx];
        float3 cube_v[12];
        for (uint32_t i = 0; i < 12; i++) {
            if (edge_status & (1 << i)) {
                int p_0 = edges[i * 2];
                int p_1 = edges[i * 2 + 1];
                cube_v[i] = interpolate(level, p_val[p_0], p_val[p_1],
                                        p_pos[p_0], p_pos[p_1]);
            }
        }

        // Assemble the triangles.
        case_idx *= 16;
        uint32_t v0_idx = result_idx * 18;
        for (uint32_t i = 0; i < 16; i += 3) {
            uint32_t tri_idx = case_idx + i;
            uint32_t v_idx = v0_idx + i;
            if (tri_table[tri_idx] != -1) {
                const float3 &v0 = cube_v[tri_table[tri_idx + 0]];
                const float3 &v1 = cube_v[tri_table[tri_idx + 1]];
                const float3 &v2 = cube_v[tri_table[tri_idx + 2]];

                if (v0 != v1 && v0 != v2 && v1 != v2) {
                    v[v_idx + 0] = v0;
                    v[v_idx + 1] = v1;
                    v[v_idx + 2] = v2;
                }
            }
        }
    }
};

std::tuple<float *, uint32_t, int *, uint32_t>
marching_cubes(float *const grid_ptr, const std::array<int64_t, 3> &grid_shape,
               const std::array<float, 6> &aabb, float level, bool tight) {
    uint3 res = make_uint3(grid_shape[0], grid_shape[1], grid_shape[2]);
    uint32_t num_cubes = (res.x - 1) * (res.y - 1) * (res.z - 1);

    // Move the grid to the device.
    thrust::device_ptr<float> grid_dp(grid_ptr);

    // Move the LUTs to the device.
    thrust::device_vector<int> edges_dv(edges, edges + edges_size);
    thrust::device_vector<int> edge_table_dv(edge_table,
                                             edge_table + edge_table_size);
    thrust::device_vector<int> tri_table_dv(tri_table,
                                            tri_table + tri_table_size);

    // Get the case index of each cube based on the sign of cube vertices.
    thrust::device_vector<uint8_t> case_idx_dv(num_cubes);
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(num_cubes),
        get_case_idx_op(thrust::raw_pointer_cast(case_idx_dv.data()),
                        thrust::raw_pointer_cast(grid_dp), res, level, tight));

    // Remove empty cubes.
    thrust::device_vector<uint32_t> grid_idx_dv(num_cubes);
    thrust::sequence(grid_idx_dv.begin(), grid_idx_dv.end());
    grid_idx_dv.erase(thrust::remove_if(grid_idx_dv.begin(), grid_idx_dv.end(),
                                        case_idx_dv.begin(), is_empty_pred()),
                      grid_idx_dv.end());
    case_idx_dv.erase(thrust::remove_if(case_idx_dv.begin(), case_idx_dv.end(),
                                        case_idx_dv.begin(), is_empty_pred()),
                      case_idx_dv.end());
    num_cubes = grid_idx_dv.size();

    auto input_iter_b = thrust::make_zip_iterator(
        thrust::make_tuple(case_idx_dv.begin(), grid_idx_dv.begin(),
                           thrust::counting_iterator<uint32_t>(0)));
    auto input_iter_e = thrust::make_zip_iterator(
        thrust::make_tuple(case_idx_dv.end(), grid_idx_dv.end(),
                           thrust::counting_iterator<uint32_t>(num_cubes)));

    // Allocate memory for the vertex array
    thrust::device_vector<float3> v_dv(num_cubes * 18);
    thrust::fill(v_dv.begin(), v_dv.end(), make_float3(NAN, NAN, NAN));

    // Run Marching Cubes on each cube.
    thrust::for_each(
        input_iter_b, input_iter_e,
        process_cube_op(thrust::raw_pointer_cast(v_dv.data()),
                        thrust::raw_pointer_cast(grid_dp),
                        thrust::raw_pointer_cast(edges_dv.data()),
                        thrust::raw_pointer_cast(edge_table_dv.data()),
                        thrust::raw_pointer_cast(tri_table_dv.data()), res,
                        level, tight));

    // Remove unused entries, which are marked as NAN.
    v_dv.erase(thrust::remove_if(v_dv.begin(), v_dv.end(), is_nan_pred()),
               v_dv.end());

    // Weld/merge vertices.
    thrust::device_vector<int> f_dv(v_dv.size());
    thrust::sequence(f_dv.begin(), f_dv.end());
    vertex_welding(v_dv, f_dv);

    // Fit the vertices inside the aabb.
    float3 aabb_min = make_float3(aabb[0], aabb[1], aabb[2]);
    float3 aabb_max = make_float3(aabb[3], aabb[4], aabb[5]);
    float3 old_scale = make_float3(res.x - 1, res.y - 1, res.z - 1);
    thrust::transform(v_dv.begin(), v_dv.end(), v_dv.begin(),
                      transform_aabb_functor(aabb_min, aabb_max, old_scale));

    // Allocate memory for the vertex pointer and copy the data.
    uint32_t v_len = v_dv.size();
    thrust::device_ptr<float3> v_dp = thrust::device_malloc<float3>(v_len);
    thrust::copy(v_dv.begin(), v_dv.end(), v_dp);
    float *v_ptr = reinterpret_cast<float *>(thrust::raw_pointer_cast(v_dp));

    // Allocate memory for the face pointer and copy the data.
    uint32_t f_len = f_dv.size() / 3;
    thrust::device_ptr<int> f_dp = thrust::device_malloc<int>(f_dv.size());
    thrust::copy(f_dv.begin(), f_dv.end(), f_dp);
    int *f_ptr = thrust::raw_pointer_cast(f_dp);

    return std::tie(v_ptr, v_len, f_ptr, f_len);
}

}   // namespace mc::lorensen