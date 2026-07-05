#include "math.cuh"
#include "shared_luts.cuh"
#include "utils.cuh"

#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void
vertex_welding(thrust::device_vector<float3> &v,
               thrust::device_vector<int> &f) {
    // Remove duplicated vertices
    thrust::device_vector<float3> sorted_v = v;
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
