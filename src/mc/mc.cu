#include "mc/mc.cuh"
#include "utils.cuh"
#include <memory>
#include <stdexcept>

#include <thrust/copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace mc {

std::tuple<NDArray<float3>, NDArray<int>>
marching_cubes(Grid *grid, float level, std::string method) {
    uint num_cells = grid->get_num_cells();
    NDArray<float> values = grid->get_values();
    NDArray<float3> points = grid->get_points();
    NDArray<uint> cells = grid->get_cells();
    auto mc_variant = MCBase::create(method);

    // Get the case index of each cell.
    thrust::device_vector<uint8_t> cases_dv(num_cells);
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_cells),
                     get_case_num_op(cases_dv.data().get(), values.data(),
                                     cells.data(), level));

    // Remove empty cells.
    thrust::device_vector<uint32_t> cell_indices_dv(num_cells);
    thrust::sequence(cell_indices_dv.begin(), cell_indices_dv.end());
    cell_indices_dv.erase(thrust::remove_if(cell_indices_dv.begin(),
                                            cell_indices_dv.end(),
                                            cases_dv.begin(), is_empty_pred()),
                          cell_indices_dv.end());
    cases_dv.erase(thrust::remove_if(cases_dv.begin(), cases_dv.end(),
                                     cases_dv.begin(), is_empty_pred()),
                   cases_dv.end());
    num_cells = cell_indices_dv.size();

    // Allocate memory for the vertex array
    thrust::device_vector<float3> v_dv(num_cells *
                                       mc_variant->get_max_triangles() * 3);
    thrust::fill(v_dv.begin(), v_dv.end(), make_float3(NAN, NAN, NAN));

    // Run Marching Cubes on each cube.
    mc_variant->run(cases_dv, cell_indices_dv,
                    thrust::raw_pointer_cast(v_dv.data()), values.data(),
                    points.data(), cells.data(), level);

    // Remove unused entries, which are marked as NAN.
    v_dv.erase(thrust::remove_if(v_dv.begin(), v_dv.end(), is_nan_pred()),
               v_dv.end());

    // Weld/merge vertices.
    thrust::device_vector<int> f_dv(v_dv.size());
    thrust::sequence(f_dv.begin(), f_dv.end());
    vertex_welding(v_dv, f_dv);

    NDArray<float3> v = NDArray<float3>::copy(v_dv.data().get(), {v_dv.size()});
    NDArray<int> f =
        NDArray<int>::copy(f_dv.data().get(), {f_dv.size() / 3, 3});

    return {v, f};
}

}   // namespace mc
