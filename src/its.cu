#include "its.cuh"
#include "shared_luts.cuh"
#include "utils.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

struct get_its_op {
    float3 *its_points;
    uint2 *its_edges;
    bool *its_is_out;
    const uint *cell_offsets;
    const uint *cell_indices;
    const int *edge_status;
    const float *values;
    const float3 *points;
    const uint *cells;
    const uint *actual_cells;
    const int *edges_table;
    const float level;

    get_its_op(float3 *its_points, uint2 *its_edges, bool *its_is_out,
               const uint *cell_offsets, const uint *cell_indices,
               const int *edge_status, const float *values,
               const float3 *points, const uint *cells,
               const uint *actual_cells, const int *edges_table,
               const float level)
        : its_points(its_points), its_edges(its_edges), its_is_out(its_is_out),
          cell_offsets(cell_offsets), cell_indices(cell_indices),
          edge_status(edge_status), values(values), points(points),
          cells(cells), actual_cells(actual_cells), edges_table(edges_table),
          level(level) {}

    __host__ __device__ void operator()(uint idx) {
        int status = edge_status[idx];
        int offset = cell_offsets[idx];

        // Compute the location of each cube vertex.
        float3 c_p[8];
        float c_v[8];
        uint c_offset = cell_indices[idx] * 8;
        for (uint32_t i = 0; i < 8; i++) {
            c_p[i] = points[cells[c_offset + i]];
            c_v[i] = values[cells[c_offset + i]];
        }

        for (int i = 0; i < 12; i++) {
            if (status & (1 << i)) {
                // Get the two vertices that form the edge.
                int p_0 = edges_table[i * 2];
                int p_1 = edges_table[i * 2 + 1];
                its_edges[offset] = make_uint2(actual_cells[idx * 8 + p_0],
                                               actual_cells[idx * 8 + p_1]);
                its_is_out[offset] = c_v[p_0] <= c_v[p_1];

                // Compute the intersection point.
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
    uint3 shape = grid->get_shape();
    NDArray<float> values = grid->get_values();
    NDArray<float3> points = grid->get_points();
    NDArray<uint> cells = grid->get_cells();
    thrust::device_vector<int> edges_table_dv(edges_table,
                                              edges_table + edges_size);
    thrust::device_vector<int> edge_status_table_dv(
        edge_status_table, edge_status_table + edge_table_size);

    // Get the case index of each cell.
    thrust::device_vector<int> edge_status(num_cells);
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_cells),
                     get_edge_status_op(
                         edge_status.data().get(), values.data(), cells.data(),
                         edge_status_table_dv.data().get(), level));

    // Remove empty cells.
    thrust::device_vector<uint> cell_indices_dv(num_cells);
    thrust::sequence(cell_indices_dv.begin(), cell_indices_dv.end());
    cell_indices_dv.erase(
        thrust::remove_if(cell_indices_dv.begin(), cell_indices_dv.end(),
                          edge_status.begin(), is_zero_pred()),
        cell_indices_dv.end());
    edge_status.erase(thrust::remove_if(edge_status.begin(), edge_status.end(),
                                        is_zero_pred()),
                      edge_status.end());
    num_cells = cell_indices_dv.size();

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

    // Initialize the intersection struct.
    Intersection its(num_points);
    its.cell_offsets = std::move(cell_offsets);
    its.cell_indices =
        NDArray<uint>::copy(cell_indices_dv.data().get(), {num_cells});

    // Get the actual cell indices. For UniformGrid, this is the same as
    // cell_indices. But for SparseGrid, cell_indices gives the indices of the
    // cells in the sparse grid, while actual_cell_indices gives the indices of
    // the cells in the uniform grid.
    thrust::device_vector<uint> actual_cell_indices_dv =
        grid->get_cell_indices();
    its.actual_cell_indices = NDArray<uint>({num_cells});
    thrust::transform(cell_indices_dv.begin(), cell_indices_dv.end(),
                      its.actual_cell_indices.data_ptr,
                      [actual_cell_indices =
                           actual_cell_indices_dv.data()] __device__(uint idx) {
                          return actual_cell_indices[idx];
                      });

    // For SparseGrid, cells and actual_cells are different. cells is used for
    // accessing points and values, while actual_cells gives the actual cell
    // indices, treating the grid as a uniform grid.
    NDArray<uint> actual_cells({num_cells, 8});
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_cells),
                     idx_to_cell_op(actual_cells.data(),
                                    its.actual_cell_indices.data(), shape));

    // Get the intersection points.
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_cells),
        get_its_op(its.points.data(), its.edges.data(), its.is_out.data(),
                   its.cell_offsets.data(), cell_indices_dv.data().get(),
                   edge_status.data().get(), values.data(), points.data(),
                   cells.data(), actual_cells.data(),
                   edges_table_dv.data().get(), level));

    return its;
}