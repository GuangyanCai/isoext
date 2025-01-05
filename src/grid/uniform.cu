#include "grid/uniform.cuh"
#include "utils.cuh"

#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

UniformGrid::UniformGrid(uint3 shape, float3 aabb_min, float3 aabb_max,
                         float default_value)
    : shape(shape), aabb_min(aabb_min), aabb_max(aabb_max),
      num_cells((shape.x - 1) * (shape.y - 1) * (shape.z - 1)),
      num_points(shape.x * shape.y * shape.z) {
    size_t size_check = (size_t) shape.x * (size_t) shape.y * (size_t) shape.z;
    if (size_check > INT_MAX) {
        throw std::runtime_error(
            "Number of points exceeds maximum value 2147483647 (max int)");
    }
    values = NDArray<float>({shape.x, shape.y, shape.z});
    values.fill(default_value);
}

NDArray<float3>
UniformGrid::get_points() const {
    NDArray<float3> points({shape.x, shape.y, shape.z});
    thrust::transform(thrust::counting_iterator<uint>(0),
                      thrust::counting_iterator<uint>(num_points),
                      points.data_ptr,
                      get_vtx_pos_op(shape, aabb_min, aabb_max));
    return points;
}

NDArray<float>
UniformGrid::get_values() const {
    return values;
}

void
UniformGrid::set_values(const NDArray<float> &new_values) {
    values.set(new_values);
}

NDArray<uint>
UniformGrid::get_cells() const {
    NDArray<uint> cells({shape.x - 1, shape.y - 1, shape.z - 1, 8});
    thrust::device_vector<uint> cell_indices(num_cells);
    thrust::sequence(cell_indices.begin(), cell_indices.end());
    thrust::for_each(
        cell_indices.begin(), cell_indices.end(),
        idx_to_cell_op(cells.data(), cell_indices.data().get(), shape));
    return cells;
}

thrust::device_vector<uint>
UniformGrid::get_cell_indices() const {
    thrust::device_vector<uint> cell_indices(num_cells);
    thrust::sequence(cell_indices.begin(), cell_indices.end());
    return cell_indices;
}

std::tuple<thrust::device_vector<int4>, thrust::device_vector<bool>>
UniformGrid::get_dual_quads(const NDArray<uint2> &edges,
                            const NDArray<bool> &is_out) const {
    // Copy edges and is_out
    thrust::device_vector<uint2> edges_dv(edges.data(),
                                          edges.data() + edges.size());
    thrust::device_vector<bool> is_out_dv(is_out.data(),
                                          is_out.data() + is_out.size());

    // Remove duplicated edges. At the same time, update the is_out array
    // accordingly.
    thrust::sort_by_key(edges_dv.begin(), edges_dv.end(), is_out_dv.begin(),
                        uint2_less_pred());
    auto new_end = thrust::unique_by_key(edges_dv.begin(), edges_dv.end(),
                                         is_out_dv.begin(), uint2_equal_pred());
    edges_dv.erase(new_end.first, edges_dv.end());
    is_out_dv.erase(new_end.second, is_out_dv.end());

    // Get edge neighbors
    thrust::device_vector<int4> edge_neighbors_dv =
        get_edge_neighbors(edges_dv, shape);

    return {edge_neighbors_dv, is_out_dv};
}
