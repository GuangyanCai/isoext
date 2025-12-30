#include "its.cuh"
#include "math.cuh"
#include "shared_luts.cuh"
#include "utils.cuh"

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
    const int *edges_table;
    const float level;

    get_its_op(float3 *its_points, uint2 *its_edges, bool *its_is_out,
               const uint *cell_offsets, const uint *cell_indices,
               const int *edge_status, const float *values,
               const float3 *points, const uint *cells, const int *edges_table,
               const float level)
        : its_points(its_points), its_edges(its_edges), its_is_out(its_is_out),
          cell_offsets(cell_offsets), cell_indices(cell_indices),
          edge_status(edge_status), values(values), points(points),
          cells(cells), edges_table(edges_table), level(level) {}

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
                its_edges[offset] =
                    make_uint2(cells[c_offset + p_0], cells[c_offset + p_1]);
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
get_intersection(Grid *grid, float level, bool compute_normals) {
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

    // Get the intersection points.
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_cells),
        get_its_op(its.points.data(), its.edges.data(), its.is_out.data(),
                   its.cell_offsets.data(), cell_indices_dv.data().get(),
                   edge_status.data().get(), values.data(), points.data(),
                   cells.data(), edges_table_dv.data().get(), level));

    // Optionally compute normals
    if (compute_normals) {
        compute_intersection_normals(its, grid);
        its._has_normals = true;
    }

    return its;
}

// Cube corner ordering (from idx_to_cell_op in utils.cuh):
// Corner index = x*4 + y*2 + z where x,y,z are 0 or 1
//   0: (0,0,0)  1: (0,0,1)  2: (0,1,0)  3: (0,1,1)
//   4: (1,0,0)  5: (1,0,1)  6: (1,1,0)  7: (1,1,1)

namespace {

// Trilinear interpolation of a value at position t within a cell
// t is in [0,1]^3, c_v are the 8 corner values
__host__ __device__ float
trilinear_interp(float3 t, const float *c_v) {
    // Interpolate along z first
    float c00 = c_v[0] * (1 - t.z) + c_v[1] * t.z;  // x=0, y=0
    float c01 = c_v[2] * (1 - t.z) + c_v[3] * t.z;  // x=0, y=1
    float c10 = c_v[4] * (1 - t.z) + c_v[5] * t.z;  // x=1, y=0
    float c11 = c_v[6] * (1 - t.z) + c_v[7] * t.z;  // x=1, y=1

    // Interpolate along y
    float c0 = c00 * (1 - t.y) + c01 * t.y;  // x=0
    float c1 = c10 * (1 - t.y) + c11 * t.y;  // x=1

    // Interpolate along x
    return c0 * (1 - t.x) + c1 * t.x;
}

struct compute_normals_op {
    float3 *normals;
    const float3 *its_points;
    const uint *cell_offsets;
    const uint *cell_indices;
    const float *values;
    const float3 *grid_points;
    const uint *cells;

    compute_normals_op(float3 *normals, const float3 *its_points,
                       const uint *cell_offsets, const uint *cell_indices,
                       const float *values, const float3 *grid_points,
                       const uint *cells)
        : normals(normals), its_points(its_points), cell_offsets(cell_offsets),
          cell_indices(cell_indices), values(values), grid_points(grid_points),
          cells(cells) {}

    __host__ __device__ void operator()(uint cell_idx) {
        uint offset = cell_offsets[cell_idx];
        uint next_offset = cell_offsets[cell_idx + 1];

        // Get the 8 corner values and positions
        uint c_offset = cell_indices[cell_idx] * 8;
        float c_v[8];
        float3 c_p[8];
        for (uint i = 0; i < 8; i++) {
            c_p[i] = grid_points[cells[c_offset + i]];
            c_v[i] = values[cells[c_offset + i]];
        }

        // Cell origin and size
        // Corner 0 is at (min_x, min_y, min_z), corner 7 is at (max_x, max_y, max_z)
        float3 cell_min = c_p[0];
        float3 cell_size = c_p[7] - c_p[0];

        // Process each intersection point in this cell
        for (uint i = offset; i < next_offset; i++) {
            float3 p = its_points[i];

            // Compute normalized position within cell [0,1]^3
            float3 t = (p - cell_min) / cell_size;

            // Clamp to valid range
            t.x = fmaxf(0.01f, fminf(0.99f, t.x));
            t.y = fmaxf(0.01f, fminf(0.99f, t.y));
            t.z = fmaxf(0.01f, fminf(0.99f, t.z));

            // Compute gradient using central differences
            // Sample at p ± epsilon in each direction
            float eps = 0.02f;   // Small offset in normalized coords

            float3 t_px = make_float3(fminf(t.x + eps, 0.99f), t.y, t.z);
            float3 t_mx = make_float3(fmaxf(t.x - eps, 0.01f), t.y, t.z);
            float3 t_py = make_float3(t.x, fminf(t.y + eps, 0.99f), t.z);
            float3 t_my = make_float3(t.x, fmaxf(t.y - eps, 0.01f), t.z);
            float3 t_pz = make_float3(t.x, t.y, fminf(t.z + eps, 0.99f));
            float3 t_mz = make_float3(t.x, t.y, fmaxf(t.z - eps, 0.01f));

            float dfdx = (trilinear_interp(t_px, c_v) -
                          trilinear_interp(t_mx, c_v)) /
                         ((t_px.x - t_mx.x) * cell_size.x);
            float dfdy = (trilinear_interp(t_py, c_v) -
                          trilinear_interp(t_my, c_v)) /
                         ((t_py.y - t_my.y) * cell_size.y);
            float dfdz = (trilinear_interp(t_pz, c_v) -
                          trilinear_interp(t_mz, c_v)) /
                         ((t_pz.z - t_mz.z) * cell_size.z);

            float3 grad = make_float3(dfdx, dfdy, dfdz);

            // Normalize to get the normal
            float grad_len = norm(grad);
            float3 n;
            if (grad_len > 1e-8f) {
                n = grad / grad_len;
            } else {
                n = make_float3(0.0f, 0.0f, 1.0f);   // fallback
            }

            normals[i] = n;
        }
    }
};
}   // anonymous namespace

void
compute_intersection_normals(Intersection &its, Grid *grid) {
    NDArray<float> values = grid->get_values();
    NDArray<float3> grid_points = grid->get_points();
    NDArray<uint> cells = grid->get_cells();

    uint num_cells = its.cell_indices.size();
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_cells),
        compute_normals_op(its.normals.data(), its.points.data(),
                           its.cell_offsets.data(), its.cell_indices.data(),
                           values.data(), grid_points.data(), cells.data()));
}