#pragma once

#include "base.cuh"
#include "grid/grid.cuh"
#include "lorensen.cuh"
#include "nagae.cuh"
#include "ndarray.cuh"

#include <string>
#include <tuple>

namespace mc {

std::tuple<NDArray<float3>, NDArray<int>>
marching_cubes(Grid *grid, float level, std::string method);

}   // namespace mc