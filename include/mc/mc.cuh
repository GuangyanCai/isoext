#pragma once

#include "base.cuh"
#include "lorensen.cuh"
#include "nagae.cuh"

#include <string>
#include <tuple>

namespace mc {

std::tuple<float *, uint32_t, int *, uint32_t>
marching_cubes(const float *grid_ptr, const float3 *cells_ptr, uint3 res,
               float level, bool tight, std::string method);
}
