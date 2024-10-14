#pragma once

#include <array>
#include <cstdint>
#include <tuple>

namespace mc {
namespace lorensen {
std::tuple<float *, uint32_t, int *, uint32_t>
marching_cubes(float *const grid_ptr, uint3 res, float3 aabb_min,
               float3 aabb_max, float level, bool tight);
}
}   // namespace mc