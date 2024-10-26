#include "lorensen.cuh"
#include "nagae.cuh"
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
    std::string, std::function<void(const thrust::device_vector<uint8_t> &,
                                    const thrust::device_vector<uint32_t> &,
                                    float3 *, const float *, const float3 *,
                                    const uint3, float, bool)>>
    method_map = {{"lorensen", lorensen::run}, {"nagae", nagae::run}};

std::tuple<float *, uint32_t, int *, uint32_t>
marching_cubes(const float *grid_ptr, uint3 res,
               std::optional<std::array<float, 6>> o_aabb,
               std::optional<const float3 *> o_cells_ptr, float level,
               std::string method) {

    auto it = method_map.find(method);
    if (it == method_map.end()) {
        throw std::runtime_error("Unknown method: " + method);
    }
    auto mc_method = it->second;

    // Variables to store cube and cell information
    uint32_t num_cubes;
    const float3 *cells_ptr;
    thrust::device_vector<float3> cells_dv;
    bool tight;

    // Check if AABB (Axis-Aligned Bounding Box) is provided
    if (o_aabb.has_value()) {
        // Ensure that cell positions are not provided when AABB is given
        if (o_cells_ptr.has_value()) {
            throw std::runtime_error(
                "Only one of AABB and cell positions is required.");
        }

        // Check if resolution is at least (2, 2, 2) for AABB
        if (res.x < 2 || res.y < 2 || res.z < 2) {
            throw std::runtime_error(
                "When given AABB, res must be at least (2, 2, 2).");
        }

        // Extract AABB values and create min/max vectors
        auto aabb = o_aabb.value();
        float3 aabb_min = make_float3(aabb[0], aabb[1], aabb[2]);
        float3 aabb_max = make_float3(aabb[3], aabb[4], aabb[5]);

        // Calculate number of cubes and points
        num_cubes = (res.x - 1) * (res.y - 1) * (res.z - 1);
        uint32_t num_points = res.x * res.y * res.z;

        // Resize cells vector and populate it with vertex positions
        cells_dv.resize(num_points);
        thrust::transform(thrust::counting_iterator<uint32_t>(0),
                          thrust::counting_iterator<uint32_t>(num_points),
                          cells_dv.begin(),
                          get_vtx_pos_op(res, aabb_min, aabb_max));
        cells_ptr = thrust::raw_pointer_cast(cells_dv.data());
        tight = true;
    }
    // Check if cell positions are provided
    else if (o_cells_ptr.has_value()) {
        // Ensure resolution is (2n, 2, 2) when cell positions are given
        if (res.x % 2 != 0 || res.y != 2 || res.z != 2) {
            throw std::runtime_error(
                "When given cell positions, res must be (2n, 2, 2).");
        }
        cells_ptr = o_cells_ptr.value();
        num_cubes = res.x / 2;
        tight = false;
    }
    // Throw error if neither AABB nor cell positions are provided
    else {
        throw std::runtime_error("Either AABB or cell positions are required.");
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
    thrust::device_vector<float3> v_dv(num_cubes * 18);
    thrust::fill(v_dv.begin(), v_dv.end(), make_float3(NAN, NAN, NAN));

    // Run Marching Cubes on each cube.
    mc_method(case_idx_dv, grid_idx_dv, thrust::raw_pointer_cast(v_dv.data()),
              grid_ptr, cells_ptr, res, level, tight);

    // Remove unused entries, which are marked as NAN.
    v_dv.erase(thrust::remove_if(v_dv.begin(), v_dv.end(), is_nan_pred()),
               v_dv.end());

    // Weld/merge vertices.
    thrust::device_vector<int> f_dv(v_dv.size());
    thrust::sequence(f_dv.begin(), f_dv.end());
    vertex_welding(v_dv, f_dv);

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
