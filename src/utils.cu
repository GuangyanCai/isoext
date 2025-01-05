#include "math.cuh"
#include "shared_luts.cuh"
#include "utils.cuh"

#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

__host__ __device__ uint3
idx_1d_to_3d(uint idx, uint3 shape) {
    uint z = idx % shape.z;
    idx /= shape.z;
    uint y = idx % shape.y;
    idx /= shape.y;
    uint x = idx;
    return make_uint3(x, y, z);
}

__device__ __host__ uint
idx_3d_to_1d(uint3 idx, uint3 shape) {
    return idx.x * shape.y * shape.z + idx.y * shape.z + idx.z;
}

__device__ __host__ uint
point_idx_to_cell_idx(uint idx, uint3 shape) {
    uint3 idx_3d = idx_1d_to_3d(idx, shape);
    return idx_3d_to_1d(idx_3d, shape - 1);
}

void
vertex_welding(thrust::device_vector<float3> &v, thrust::device_vector<int> &f,
               bool skip_scatter) {

    thrust::device_vector<float3> sorted_v;

    if (skip_scatter) {
        sorted_v = v;
    } else {
        // Scatter v to sorted_v based on f
        thrust::scatter(v.begin(), v.end(), f.begin(), sorted_v.begin());
        f.clear();
        f.resize(v.size());
        thrust::sequence(f.begin(), f.end());
    }

    // Remove duplicated vertices
    thrust::sort(sorted_v.begin(), sorted_v.end(), float3_less_pred());
    sorted_v.erase(
        thrust::unique(sorted_v.begin(), sorted_v.end(), float3_elem_eq_pred()),
        sorted_v.end());

    thrust::lower_bound(sorted_v.begin(), sorted_v.end(), v.begin(), v.end(),
                        f.begin(), float3_less_pred());

    // Update vertex array
    v = std::move(sorted_v);
}

thrust::device_vector<int4>
get_edge_neighbors(const thrust::device_vector<uint2> &edges_dv,
                   uint3 grid_shape) {
    thrust::device_vector<uint3> en_table(
        edge_neighbors_table, edge_neighbors_table + edge_neighbors_table_size);

    thrust::device_vector<int4> edge_neighbors_dv(edges_dv.size());
    thrust::transform(
        edges_dv.begin(), edges_dv.end(), edge_neighbors_dv.begin(),
        edge_to_neighbor_idx_op(en_table.data().get(), grid_shape));

    return edge_neighbors_dv;
}
