#include "its.cuh"
#include "shared_luts.cuh"

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace {
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
}   // anonymous namespace

Intersection
get_intersection(Grid *grid, float level) {
    uint num_cells = grid->get_num_cells();
    NDArray<float> values = grid->get_values();
    NDArray<float3> points = grid->get_points();
    NDArray<uint> cells = grid->get_cells();
    thrust::device_vector<int> edges_dv(edges, edges + edges_size);
    thrust::device_vector<int> edge_table_dv(edge_table,
                                             edge_table + edge_table_size);

    // Get the case index of each cell.
    thrust::device_vector<int> edge_status(num_cells);
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_cells),
                     get_edge_status_op(edge_status.data().get(), values.data(),
                                        cells.data(),
                                        edge_table_dv.data().get(), level));

    // Remove empty cells.
    thrust::device_vector<uint> cell_idx_dv(num_cells);
    thrust::sequence(cell_idx_dv.begin(), cell_idx_dv.end());
    cell_idx_dv.erase(thrust::remove_if(cell_idx_dv.begin(), cell_idx_dv.end(),
                                        edge_status.begin(), is_zero_pred()),
                      cell_idx_dv.end());
    edge_status.erase(thrust::remove_if(edge_status.begin(), edge_status.end(),
                                        is_zero_pred()),
                      edge_status.end());
    num_cells = cell_idx_dv.size();

    // Compute the number of intersections.
    NDArray<uint> cell_offsets({num_cells + 1});
    thrust::transform(edge_status.begin(), edge_status.end(),
                      cell_offsets.data_ptr,
                      [=] __device__(int es) { return __popc(es); });

    // Compute the prefix sum of the number of intersections.
    thrust::exclusive_scan(cell_offsets.data_ptr,
                           cell_offsets.data_ptr + num_cells + 1,
                           cell_offsets.data_ptr, 0);
    uint num_points = cell_offsets.data_ptr[num_cells];

    // Get the intersection points.
    Intersection its(num_points, std::move(cell_offsets));
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_cells),
        get_its_points_op(its.points.data(), cell_idx_dv.data().get(),
                          its.cell_offsets.data(), edge_status.data().get(),
                          values.data(), points.data(), cells.data(),
                          edges_dv.data().get(), level));

    return its;
}