#include "mc/nagae.cuh"
#include "shared_luts.cuh"
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace mc {

namespace {
static MCRegistrar<Nagae> registrar("nagae");

struct process_cube_op {
    float3 *v;
    const float *values;
    const float3 *points;
    const uint *cells;
    const int *edges;
    const int *edge_table;
    const int *tri_table;
    const float level;

    process_cube_op(float3 *v, const float *values, const float3 *points,
                    const uint *cells, const int *edges, const int *edge_table,
                    const int *tri_table, float level)
        : v(v), values(values), points(points), cells(cells), edges(edges),
          edge_table(edge_table), tri_table(tri_table), level(level) {}

    __host__ __device__ void
    operator()(thrust::tuple<uint8_t, uint32_t, uint32_t> args) {
        uint32_t case_num, cell_idx, result_idx;
        thrust::tie(case_num, cell_idx, result_idx) = args;

        // Compute the location of each cube vertex.
        float3 c_p[8];
        float c_v[8];
        uint offset = cell_idx * 8;
        for (uint32_t i = 0; i < 8; i++) {
            c_p[i] = points[cells[offset + i]];
            c_v[i] = values[cells[offset + i]];
        }

        // Compute the intersection between the isosurface and each edge.
        int edge_status = edge_table[case_num];
        float3 cube_v[12];
        for (uint32_t i = 0; i < 12; i++) {
            if (edge_status & (1 << i)) {
                int p_0 = edges[i * 2];
                int p_1 = edges[i * 2 + 1];
                float denom = c_v[p_1] - c_v[p_0];
                float t = (denom != 0.0f) ? (level - c_v[p_0]) / denom : 0.0f;
                cube_v[i] = lerp(t, c_p[p_0], c_p[p_1]);
            }
        }

        // Assemble the triangles.
        case_num *= Nagae::max_len;   // max_length = 3 * 5 for Nagae
        uint32_t v0_idx = result_idx * Nagae::max_len;
        for (uint32_t i = 0; i < Nagae::max_len; i += 3) {
            uint32_t tri_idx = case_num + i;
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
Nagae::run(const thrust::device_vector<uint8_t> &case_num_dv,
           const thrust::device_vector<uint> &cell_indices_dv, float3 *v,
           const float *values, const float3 *points, const uint *cells,
           float level) {
    // Move the LUTs to the device.
    thrust::device_vector<int> edges_dv(edges, edges + edges_size);
    thrust::device_vector<int> edge_table_dv(edge_table,
                                             edge_table + edge_table_size);
    thrust::device_vector<int> tri_table_dv(
        Nagae::tri_table, Nagae::tri_table + Nagae::tri_table_size);

    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(case_num_dv.begin(), cell_indices_dv.begin(),
                           thrust::counting_iterator<uint32_t>(0)));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(
        case_num_dv.end(), cell_indices_dv.end(),
        thrust::counting_iterator<uint32_t>(case_num_dv.size())));

    thrust::for_each(begin, end,
                     process_cube_op(v, values, points, cells,
                                     edges_dv.data().get(),
                                     edge_table_dv.data().get(),
                                     tri_table_dv.data().get(), level));
}

}   // namespace mc