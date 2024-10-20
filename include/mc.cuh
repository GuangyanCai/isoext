#pragma once

#include <array>
#include <optional>
#include <string>
#include <tuple>

namespace mc {

std::tuple<float *, uint32_t, int *, uint32_t>
marching_cubes(const float *grid_ptr, uint3 res,
               std::optional<std::array<float, 6>> o_aabb,
               std::optional<const float3 *> o_cells_ptr, float level,
               std::string method);
}
