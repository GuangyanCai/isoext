#pragma once

#include "grid/grid.cuh"
#include "its.cuh"
#include "ndarray.cuh"

#include <string>
#include <tuple>

std::tuple<NDArray<float3>, NDArray<int>>
dual_marching_cubes(Grid *grid, const Intersection &its, float level = 0.0f,
                    std::string method = "vega");
