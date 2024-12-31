#include "math.cuh"
#include "utils.cuh"

#include <thrust/binary_search.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <iostream>

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
