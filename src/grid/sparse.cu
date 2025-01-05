#include "grid/sparse.cuh"
#include "utils.cuh"

#include <thrust/remove.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

SparseGrid::SparseGrid(uint3 shape, float3 aabb_min, float3 aabb_max,
                       float default_value)
    : shape(shape), aabb_min(aabb_min), aabb_max(aabb_max),
      default_value(default_value) {
    size_t size_check = (size_t) shape.x * (size_t) shape.y * (size_t) shape.z;
    if (size_check > INT_MAX) {
        throw std::runtime_error(
            "Maximum number of points exceeds maximum value 2147483647 "
            "(max int)");
    }
}

NDArray<float3>
SparseGrid::get_points_from_cell_indices(
    const NDArray<uint> &cell_indices_) const {
    uint num_cells = cell_indices_.size();

    NDArray<uint> cells({num_cells, 8});
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_cells),
                     idx_to_cell_op(cells.data(), cell_indices_.data(), shape));

    NDArray<float3> points({num_cells, 8});
    thrust::transform(cells.data_ptr, cells.data_ptr + cells.size(),
                      points.data_ptr,
                      get_vtx_pos_op(shape, aabb_min, aabb_max));
    return points;
}

NDArray<float3>
SparseGrid::get_points() const {
    NDArray<uint> cell_indices_ =
        NDArray<uint>::copy(cell_indices.data().get(), {cell_indices.size()});
    return get_points_from_cell_indices(cell_indices_);
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
    uint num_cells = cell_indices.size();
    thrust::device_vector<uint> cells_dv(num_cells * 8);
    thrust::sequence(cells_dv.begin(), cells_dv.end());
    NDArray<uint> cells =
        NDArray<uint>::copy(cells_dv.data().get(), {num_cells, 8});
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

thrust::device_vector<uint>
SparseGrid::get_cell_indices() const {
    return cell_indices;
}

std::vector<NDArray<int>>
SparseGrid::get_potential_cell_indices(uint chunk_size) const {
    uint num_cells = shape.x * shape.y * shape.z;
    uint num_chunks = (num_cells + chunk_size - 1) / chunk_size;
    std::vector<NDArray<int>> potential_cell_indices(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        uint start = i * chunk_size;
        uint end = std::min(start + chunk_size, num_cells);
        uint size = end - start;
        potential_cell_indices[i] = NDArray<int>({size});
        thrust::sequence(potential_cell_indices[i].data_ptr,
                         potential_cell_indices[i].data_ptr + size, start);
    }
    return potential_cell_indices;
}

NDArray<float3>
SparseGrid::get_points_by_cell_indices(
    const NDArray<uint> &new_cell_indices) const {
    return get_points_from_cell_indices(new_cell_indices);
}

NDArray<uint>
SparseGrid::filter_cell_indices(const NDArray<uint> &new_cell_indices,
                                const NDArray<float> &new_values,
                                float level) const {
    uint num_cells = new_cell_indices.size();

    // Convert cell indices to cells
    thrust::device_vector<uint> cells_dv(num_cells * 8);
    thrust::sequence(cells_dv.begin(), cells_dv.end());

    // Get the case index of each cell.
    thrust::device_vector<uint8_t> cases_dv(num_cells);
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_cells),
                     get_case_num_op(cases_dv.data().get(), new_values.data(),
                                     cells_dv.data().get(), level));
    // Remove empty cells.
    thrust::device_vector<uint> cell_indices_dv(
        new_cell_indices.data_ptr, new_cell_indices.data_ptr + num_cells);
    cell_indices_dv.erase(thrust::remove_if(cell_indices_dv.begin(),
                                            cell_indices_dv.end(),
                                            cases_dv.begin(), is_empty_pred()),
                          cell_indices_dv.end());
    num_cells = cell_indices_dv.size();

    // Convert cell indices back to cell indices
    NDArray<uint> filtered_cell_indices =
        NDArray<uint>::copy(cell_indices_dv.data().get(), {num_cells});
    return filtered_cell_indices;
}

void
SparseGrid::convert_edges(thrust::device_vector<uint2> &edges_dv,
                          thrust::device_vector<int4> &edge_neighbors_dv) {
    uint num_cells = shape.x * shape.y * shape.z;
    thrust::device_vector<int> idx_map_dv(num_cells, -1);
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(cell_indices.size()),
        [idx_map = idx_map_dv.data().get(),
         cell_indices = cell_indices.data().get()] __device__(uint idx) {
            idx_map[cell_indices[idx]] = idx;
        });

    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(edge_neighbors_dv.size()),
                     [edge_neighbors = edge_neighbors_dv.data().get(),
                      idx_map = idx_map_dv.data().get()] __device__(uint idx) {
                         int4 &en = edge_neighbors[idx];
                         en.x = idx_map[en.x];
                         en.y = idx_map[en.y];
                         en.z = idx_map[en.z];
                         en.w = idx_map[en.w];
                     });
}
