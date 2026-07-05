#include "extraction.cuh"
#include "mc/mc.cuh"

#include <thrust/device_vector.h>
#include <thrust/fill.h>

namespace mc {

std::tuple<NDArray<float3>, NDArray<int>>
marching_cubes(Grid *grid, float level, std::string method) {
    GridView view = grid->get_view();
    auto mc_variant = MCBase::create(method);

    thrust::device_vector<uint8_t> cases =
        compute_cell_cases(view, grid->get_num_cells(), level);
    thrust::device_vector<uint> cell_indices = compact_active_cells(cases);
    uint num_cells = cell_indices.size();

    // Triangle soup with unused slots marked as NAN.
    thrust::device_vector<float3> v(num_cells *
                                    mc_variant->get_max_triangles() * 3);
    thrust::fill(v.begin(), v.end(), make_float3(NAN, NAN, NAN));

    mc_variant->run(v.data().get(), num_cells, cases.data().get(),
                    cell_indices.data().get(), view, level);

    auto [v_out, f_out] = soup_to_mesh(v);
    return {v_out, f_out};
}

}   // namespace mc
