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
    thrust::for_each(cell_idx.begin(), cell_idx.end(),
                     idx_to_cell_op(cells.data_ptr.get(), shape));
    return cells;
}

void
SparseGrid::add_cells(const NDArray<uint> &new_cell_idx) {
    // Get current size of cells_idx
    uint old_size = cell_idx.size();
    uint new_size = new_cell_idx.size();

    // Resize cells_idx to fit new elements
    cell_idx.resize(old_size + new_size);

    // Copy new indices to end of cells_idx
    thrust::copy(new_cell_idx.data_ptr, new_cell_idx.data_ptr + new_size,
                 cell_idx.begin() + old_size);

    // Sort to prepare for unique
    thrust::sort(cell_idx.begin(), cell_idx.end());

    // Remove duplicates
    auto new_end = thrust::unique(cell_idx.begin(), cell_idx.end());
    cell_idx.resize(new_end - cell_idx.begin());

    // Resize values to match new number of cells (8 values per cell) and fill
    // with default value
    values.resize(cell_idx.size() * 8);
    thrust::fill(values.begin(), values.end(), default_value);
}

void
SparseGrid::remove_cells(const NDArray<uint> &new_cell_idx_) {
    // Create temporary vector for set difference operation
    thrust::device_vector<uint> result(cell_idx.size());
    thrust::device_vector<uint> new_cell_idx(new_cell_idx_.size());
    thrust::copy(new_cell_idx_.data_ptr,
                 new_cell_idx_.data_ptr + new_cell_idx_.size(),
                 new_cell_idx.begin());
    thrust::sort(new_cell_idx.begin(), new_cell_idx.end());
    auto result_end = thrust::set_difference(
        cell_idx.begin(), cell_idx.end(), new_cell_idx.begin(),
        new_cell_idx.end(), result.begin());

    // Resize result vector to actual size after set_difference
    result.resize(result_end - result.begin());

    // Copy result back to cells_idx
    cell_idx = result;

    // Resize values to match new number of cells (8 values per cell) and fill
    // with default value
    values.resize(cell_idx.size() * 8);
    thrust::fill(values.begin(), values.end(), default_value);
}

NDArray<uint>
SparseGrid::get_cell_indices() const {
    return NDArray<uint>::copy(cell_idx.data().get(), {cell_idx.size()});
}
