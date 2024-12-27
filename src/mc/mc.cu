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

std::tuple<float *, uint32_t, int *, uint32_t>
marching_cubes(const float *grid_ptr, const float3 *cells_ptr, uint3 res,
               float level, bool tight, std::string method) {

    auto mc_variant = MCBase::create(method);

    // Variables to store cube and cell information
    uint32_t num_cubes;

    if (tight) {
        // Check if resolution is at least (2, 2, 2)
        if (res.x < 2 || res.y < 2 || res.z < 2) {
            throw std::runtime_error("When the grid layout is tight, res must "
                                     "be at least (2, 2, 2).");
        }
        num_cubes = (res.x - 1) * (res.y - 1) * (res.z - 1);
    } else {
        // Ensure resolution is (2n, 2, 2) when cell positions are given
        if (res.x % 2 != 0 || res.y != 2 || res.z != 2) {
            throw std::runtime_error(
                "When the grid layout is not tight, res must be (2n, 2, 2).");
        }
        num_cubes = res.x / 2;
    }

    // Get the case index of each cube based on the sign of cube vertices.
    thrust::device_vector<uint8_t> case_idx_dv(num_cubes);
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(num_cubes),
        get_case_idx_op(thrust::raw_pointer_cast(case_idx_dv.data()), grid_ptr,
                        res, level, tight));

    // Remove empty cubes.
    thrust::device_vector<uint32_t> grid_idx_dv(num_cubes);
    thrust::sequence(grid_idx_dv.begin(), grid_idx_dv.end());
    grid_idx_dv.erase(thrust::remove_if(grid_idx_dv.begin(), grid_idx_dv.end(),
                                        case_idx_dv.begin(), is_empty_pred()),
                      grid_idx_dv.end());
    case_idx_dv.erase(thrust::remove_if(case_idx_dv.begin(), case_idx_dv.end(),
                                        case_idx_dv.begin(), is_empty_pred()),
                      case_idx_dv.end());
    num_cubes = grid_idx_dv.size();

    // Allocate memory for the vertex array
    thrust::device_vector<float3> v_dv(num_cubes *
                                       mc_variant->get_max_triangles() * 3);
    thrust::fill(v_dv.begin(), v_dv.end(), make_float3(NAN, NAN, NAN));

    // Run Marching Cubes on each cube.
    mc_variant->run(case_idx_dv, grid_idx_dv,
                    thrust::raw_pointer_cast(v_dv.data()), grid_ptr, cells_ptr,
                    res, level, tight);

    // Remove unused entries, which are marked as NAN.
    v_dv.erase(thrust::remove_if(v_dv.begin(), v_dv.end(), is_nan_pred()),
               v_dv.end());

    // Weld/merge vertices.
    thrust::device_vector<int> f_dv(v_dv.size());
    thrust::sequence(f_dv.begin(), f_dv.end());
    vertex_welding(v_dv, f_dv);

    // Allocate memory for the vertex pointer and copy the data.
    uint32_t v_len = v_dv.size();
    thrust::device_ptr<float3> v_ptr = thrust::device_malloc<float3>(v_len);
    thrust::copy(v_dv.begin(), v_dv.end(), v_ptr);
    float *v_ptr_raw =
        reinterpret_cast<float *>(thrust::raw_pointer_cast(v_ptr));

    // Allocate memory for the face pointer and copy the data.
    uint32_t f_len = f_dv.size() / 3;
    thrust::device_ptr<int> f_ptr = thrust::device_malloc<int>(f_dv.size());
    thrust::copy(f_dv.begin(), f_dv.end(), f_ptr);
    int *f_ptr_raw = thrust::raw_pointer_cast(f_ptr);

    return {v_ptr_raw, v_len, f_ptr_raw, f_len};
}

}   // namespace mc
