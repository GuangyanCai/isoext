#include "math.cuh"
#include "mc_nagae.cuh"
#include "utils.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace mc {

namespace {
static MCRegistrar<Nagae> registrar("nagae");

struct process_cube_op {
    float3 *v;
    const float *grid;
    const float3 *cells;
    const int *edges;
    const int *edge_table;
    const int *tri_table;
    const uint3 res;
    const float level;
    const bool tight;

    process_cube_op(float3 *v, const float *grid, const float3 *cells,
                    const int *edges, const int *edge_table,
                    const int *tri_table, const uint3 res, float level,
                    bool tight)
        : v(v), grid(grid), cells(cells), edges(edges), edge_table(edge_table),
          tri_table(tri_table), res(res), level(level), tight(tight) {}

    __host__ __device__ void
    operator()(thrust::tuple<uint8_t, uint32_t, uint32_t> args) {
        uint32_t case_idx, cube_idx, result_idx;
        thrust::tie(case_idx, cube_idx, result_idx) = args;

        // For each cube vertex, compute the index to the grid array.
        Cube c(cube_idx, res, tight);

        // Compute the location of each cube vertex.
        float3 p_pos[8];
        float p_val[8];
        for (uint32_t i = 0; i < 8; i++) {
            p_pos[i] = cells[c.vi[i]];
            p_val[i] = grid[c.vi[i]];
        }

        // Compute the intersection between the isosurface and each edge.
        int edge_status = edge_table[case_idx];
        float3 cube_v[12];
        for (uint32_t i = 0; i < 12; i++) {
            if (edge_status & (1 << i)) {
                int p_0 = edges[i * 2];
                int p_1 = edges[i * 2 + 1];
                float denom = p_val[p_1] - p_val[p_0];
                float t = (denom != 0.0f) ? (level - p_val[p_0]) / denom : 0.0f;
                cube_v[i] = lerp(t, p_pos[p_0], p_pos[p_1]);
            }
        }

        // Assemble the triangles.
        case_idx *= Nagae::max_len;   // max_length = 3 * 5 for Nagae
        uint32_t v0_idx = result_idx * Nagae::max_len;
        for (uint32_t i = 0; i < Nagae::max_len; i += 3) {
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
            } else {
                break;
            }
        }
    }
};
}   // anonymous namespace

void
Nagae::run(const thrust::device_vector<uint8_t> &case_idx_dv,
           const thrust::device_vector<uint32_t> &grid_idx_dv, float3 *v,
           const float *grid, const float3 *cells, const uint3 res, float level,
           bool tight) {
    // Move the LUTs to the device.
    static const thrust::device_vector<int> edges_dv(
        Nagae::edges, Nagae::edges + Nagae::edges_size);
    static const thrust::device_vector<int> edge_table_dv(
        Nagae::edge_table, Nagae::edge_table + Nagae::edge_table_size);
    static const thrust::device_vector<int> tri_table_dv(
        Nagae::tri_table, Nagae::tri_table + Nagae::tri_table_size);

    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(case_idx_dv.begin(), grid_idx_dv.begin(),
                           thrust::counting_iterator<uint32_t>(0)));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(
        case_idx_dv.end(), grid_idx_dv.end(),
        thrust::counting_iterator<uint32_t>(case_idx_dv.size())));

    thrust::for_each(
        begin, end,
        process_cube_op(
            v, grid, cells, thrust::raw_pointer_cast(edges_dv.data()),
            thrust::raw_pointer_cast(edge_table_dv.data()),
            thrust::raw_pointer_cast(tri_table_dv.data()), res, level, tight));
}

}   // namespace mc