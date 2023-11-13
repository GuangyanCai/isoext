#pragma once

#include <array>
#include <tuple>
#include <cstdint>

namespace mc {
    namespace lorensen {
        std::tuple<float *, uint32_t, int *, uint32_t> marching_cubes(
            float * const grid_ptr,  
            const std::array<int64_t, 3>& grid_shape, 
            const std::array<float, 6>& aabb, 
            float level);
    }
}