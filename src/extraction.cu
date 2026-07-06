#include "extraction.cuh"
#include "math.cuh"
#include "utils.cuh"

#include <thrust/remove.h>
#include <thrust/sequence.h>

thrust::device_vector<uint8_t>
compute_cell_cases(const GridView &view, uint num_cells, float level) {
    thrust::device_vector<uint8_t> cases(num_cells);
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_cells),
                     get_case_num_op(cases.data().get(), view, level));
    return cases;
}

thrust::device_vector<uint>
compact_active_cells(thrust::device_vector<uint8_t> &cases) {
    thrust::device_vector<uint> cell_indices(cases.size());
    thrust::sequence(cell_indices.begin(), cell_indices.end());
    cell_indices.erase(thrust::remove_if(cell_indices.begin(),
                                         cell_indices.end(), cases.begin(),
                                         is_empty_pred()),
                       cell_indices.end());
    cases.erase(thrust::remove_if(cases.begin(), cases.end(), is_empty_pred()),
                cases.end());
    return cell_indices;
}

std::pair<NDArray<float3>, NDArray<int>>
soup_to_mesh(thrust::device_vector<float3> &v) {
    // Remove unused entries, which are marked as NAN.
    v.erase(thrust::remove_if(v.begin(), v.end(), is_nan_pred()), v.end());

    // Weld/merge vertices.
    thrust::device_vector<int> f(v.size());
    thrust::sequence(f.begin(), f.end());
    vertex_welding(v, f);

    NDArray<float3> v_out = NDArray<float3>::copy(v.data().get(), {v.size()});
    NDArray<int> f_out = NDArray<int>::copy(f.data().get(), {f.size() / 3, 3});
    return {v_out, f_out};
}
