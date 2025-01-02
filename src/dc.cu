#include "dc.cuh"
#include "math.cuh"
#include "utils.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace {
struct get_triangles_op {
    float3 *v;
    const float3 *dual_v;
    const int4 *quad_indices;
    const uint2 *its_edges;
    const float *values;
    const int *idx_map;

    get_triangles_op(float3 *v, const float3 *dual_v, const int4 *quad_indices,
                     const uint2 *its_edges, const float *values,
                     const int *idx_map)
        : v(v), dual_v(dual_v), quad_indices(quad_indices),
          its_edges(its_edges), values(values), idx_map(idx_map) {}

    __host__ __device__ void operator()(uint idx) {
        int4 quad_idx = quad_indices[idx];
        if (quad_idx.x == -1 || quad_idx.y == -1 || quad_idx.z == -1 ||
            quad_idx.w == -1) {
            return;
        }

        quad_idx.x = idx_map[quad_idx.x];
        quad_idx.y = idx_map[quad_idx.y];
        quad_idx.z = idx_map[quad_idx.z];
        quad_idx.w = idx_map[quad_idx.w];

        // If the edge is pointing inward, swap the quad indices.
        uint2 edge = its_edges[idx];
        if (values[edge.x] > values[edge.y]) {
            quad_idx =
                make_int4(quad_idx.w, quad_idx.z, quad_idx.y, quad_idx.x);
        }

        v[idx * 6 + 0] = dual_v[quad_idx.x];
        v[idx * 6 + 1] = dual_v[quad_idx.y];
        v[idx * 6 + 2] = dual_v[quad_idx.z];
        v[idx * 6 + 3] = dual_v[quad_idx.x];
        v[idx * 6 + 4] = dual_v[quad_idx.z];
        v[idx * 6 + 5] = dual_v[quad_idx.w];
    }
};

}   // anonymous namespace

std::pair<NDArray<float3>, NDArray<int>>
dual_contouring(Grid *grid, const Intersection &its, float level) {
    thrust::device_vector<uint2> edges_dv(its.edges.data(),
                                          its.edges.data() + its.edges.size());

    // Sort edges to bring duplicates together
    thrust::sort(edges_dv.begin(), edges_dv.end(), uint2_less_pred());

    // Remove duplicates
    edges_dv.erase(
        thrust::unique(edges_dv.begin(), edges_dv.end(), uint2_equal_pred()),
        edges_dv.end());

    uint num_edges = edges_dv.size();

    // Get edge neighbors
    uint3 shape = grid->get_shape();
    thrust::device_vector<int4> edge_neighbors =
        get_edge_neighbors(edges_dv, shape);

    // Get average intersection point for each cell
    uint num_cells = its.cell_indices.size();
    thrust::device_vector<float3> its_points_avg_dv(num_cells);
    thrust::transform(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_cells), its_points_avg_dv.begin(),
        get_its_point_avg_op(its.points.data(), its.cell_offsets.data()));

    // Create index map that maps cell indices to its_points_avg_dv indices
    thrust::device_vector<int> idx_map(grid->get_num_cells(), -1);
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_cells),
        [idx_map = idx_map.data(),
         cell_indices = its.cell_indices.data()] __device__(uint i) {
            idx_map[cell_indices[i]] = i;
        });

    const NDArray<float> &values = grid->get_values();
    thrust::device_vector<float3> v_dv(num_edges * 6,
                                       make_float3(NAN, NAN, NAN));
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_edges),
        get_triangles_op(v_dv.data().get(), its_points_avg_dv.data().get(),
                         edge_neighbors.data().get(), edges_dv.data().get(),
                         values.data(), idx_map.data().get()));

    // Remove unused entries, which are marked as NAN.
    v_dv.erase(thrust::remove_if(v_dv.begin(), v_dv.end(), is_nan_pred()),
               v_dv.end());

    // Weld/merge vertices.
    thrust::device_vector<int> f_dv(v_dv.size());
    thrust::sequence(f_dv.begin(), f_dv.end());
    // vertex_welding(v_dv, f_dv);

    NDArray<float3> v = NDArray<float3>::copy(v_dv.data().get(), {v_dv.size()});
    NDArray<int> f =
        NDArray<int>::copy(f_dv.data().get(), {f_dv.size() / 3, 3});

    return {v, f};
}
