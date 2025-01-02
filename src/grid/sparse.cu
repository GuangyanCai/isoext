#include "grid/sparse.cuh"
#include "utils.cuh"

#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

SparseGrid::SparseGrid(uint3 shape, float3 aabb_min, float3 aabb_max,
                       float default_value)
    : shape(shape), aabb_min(aabb_min), aabb_max(aabb_max),
      default_value(default_value) {}

NDArray<float3>
SparseGrid::get_points() const {
    NDArray<float3> points({get_num_cells(), 8});
    NDArray<uint> cells = get_cells();
    thrust::transform(cells.data_ptr, cells.data_ptr + cells.size(),
                      points.data_ptr,
                      get_vtx_pos_op(shape, aabb_min, aabb_max));
    return points;
}

NDArray<float>
SparseGrid::get_values() const {
    return NDArray<float>::copy(values.data().get(), {get_num_cells(), 8});
}

void
SparseGrid::set_values(const NDArray<float> &new_values) {
    uint num_points = get_num_points();
    if (new_values.size() != num_points) {
        throw std::runtime_error(
            "New values size does not match number of points");
    }
    thrust::copy(new_values.data_ptr, new_values.data_ptr + num_points,
                 values.begin());
}

NDArray<uint>
SparseGrid::get_cells() const {
    NDArray<uint> cells({get_num_cells(), 8});
    thrust::for_each(cell_indices.begin(), cell_indices.end(),
                     idx_to_cell_op(cells.data_ptr.get(), shape));
    return cells;
}

void
SparseGrid::add_cells(const NDArray<uint> &new_cell_indices) {
    // Get current size of cells_indices
    uint old_size = cell_indices.size();
    uint new_size = new_cell_indices.size();

    // Resize cells_indices to fit new elements
    cell_indices.resize(old_size + new_size);

    // Copy new indices to end of cells_indices
    thrust::copy(new_cell_indices.data_ptr,
                 new_cell_indices.data_ptr + new_size,
                 cell_indices.begin() + old_size);

    // Sort to prepare for unique
    thrust::sort(cell_indices.begin(), cell_indices.end());

    // Remove duplicates
    auto new_end = thrust::unique(cell_indices.begin(), cell_indices.end());
    cell_indices.resize(new_end - cell_indices.begin());

    // Resize values to match new number of cells (8 values per cell) and fill
    // with default value
    values.resize(cell_indices.size() * 8);
    thrust::fill(values.begin(), values.end(), default_value);
}

void
SparseGrid::remove_cells(const NDArray<uint> &new_cell_indices_) {
    // Create temporary vector for set difference operation
    thrust::device_vector<uint> result(cell_indices.size());
    thrust::device_vector<uint> new_cell_indices(new_cell_indices_.size());
    thrust::copy(new_cell_indices_.data_ptr,
                 new_cell_indices_.data_ptr + new_cell_indices_.size(),
                 new_cell_indices.begin());
    thrust::sort(new_cell_indices.begin(), new_cell_indices.end());
    auto result_end = thrust::set_difference(
        cell_indices.begin(), cell_indices.end(), new_cell_indices.begin(),
        new_cell_indices.end(), result.begin());

    // Resize result vector to actual size after set_difference
    result.resize(result_end - result.begin());

    // Copy result back to cells_indices
    cell_indices = result;

    // Resize values to match new number of cells (8 values per cell) and fill
    // with default value
    values.resize(cell_indices.size() * 8);
    thrust::fill(values.begin(), values.end(), default_value);
}

NDArray<uint>
SparseGrid::get_cell_indices() const {
    return NDArray<uint>::copy(cell_indices.data().get(),
                               {cell_indices.size()});
}
