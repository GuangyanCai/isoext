#include "lorensen.cuh"
#include "math.cuh"
#include "mc.cuh"
#include "utils.cuh"

#include <thrust/copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

#include <unordered_map>

namespace mc {

static std::unordered_map<
    std::string,
    std::function<void(const thrust::device_vector<uint8_t> &,
                       const thrust::device_vector<uint32_t> &, float3 *,
                       const float *, const uint3, float, bool)>>
    method_map = {{"lorensen", lorensen::run}};

std::tuple<float *, uint32_t, int *, uint32_t>
marching_cubes(float *const grid_ptr, uint3 res, float3 aabb_min,
               float3 aabb_max, float level, bool tight, std::string method) {
    uint32_t num_cubes;
    if (tight) {
        num_cubes = (res.x - 1) * (res.y - 1) * (res.z - 1);
    } else {
        if (res.x % 2 != 0 || res.y != 2 || res.z != 2) {
            throw std::runtime_error(
                "When tight is false, res must be (2n, 2, 2).");
        }
        num_cubes = res.x / 2;
    }

    auto it = method_map.find(method);
    if (it == method_map.end()) {
        throw std::runtime_error("Unknown method: " + method);
    }

    auto mc_method = it->second;

    // Move the grid to the device.
    thrust::device_ptr<float> grid_dp(grid_ptr);

    // Get the case index of each cube based on the sign of cube vertices.
    thrust::device_vector<uint8_t> case_idx_dv(num_cubes);
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(num_cubes),
        get_case_idx_op(thrust::raw_pointer_cast(case_idx_dv.data()),
                        thrust::raw_pointer_cast(grid_dp), res, level, tight));

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
    thrust::device_vector<float3> v_dv(num_cubes * 18);
    thrust::fill(v_dv.begin(), v_dv.end(), make_float3(NAN, NAN, NAN));

    // Run Marching Cubes on each cube.
    mc_method(case_idx_dv, grid_idx_dv, thrust::raw_pointer_cast(v_dv.data()),
              thrust::raw_pointer_cast(grid_dp), res, level, tight);

    // Remove unused entries, which are marked as NAN.
    v_dv.erase(thrust::remove_if(v_dv.begin(), v_dv.end(), is_nan_pred()),
               v_dv.end());

    // Weld/merge vertices.
    thrust::device_vector<int> f_dv(v_dv.size());
    thrust::sequence(f_dv.begin(), f_dv.end());
    vertex_welding(v_dv, f_dv);

    // Fit the vertices inside the aabb.
    float3 old_scale = make_float3(res.x - 1, res.y - 1, res.z - 1);
    thrust::transform(v_dv.begin(), v_dv.end(), v_dv.begin(),
                      transform_aabb_functor(aabb_min, aabb_max, old_scale));

    // Allocate memory for the vertex pointer and copy the data.
    uint32_t v_len = v_dv.size();
    thrust::device_ptr<float3> v_dp = thrust::device_malloc<float3>(v_len);
    thrust::copy(v_dv.begin(), v_dv.end(), v_dp);
    float *v_ptr = reinterpret_cast<float *>(thrust::raw_pointer_cast(v_dp));

    // Allocate memory for the face pointer and copy the data.
    uint32_t f_len = f_dv.size() / 3;
    thrust::device_ptr<int> f_dp = thrust::device_malloc<int>(f_dv.size());
    thrust::copy(f_dv.begin(), f_dv.end(), f_dp);
    int *f_ptr = thrust::raw_pointer_cast(f_dp);

    return {v_ptr, v_len, f_ptr, f_len};
}

}   // namespace mc
