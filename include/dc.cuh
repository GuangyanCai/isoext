#pragma once

#include "grid/grid.cuh"
#include "its.cuh"
#include "ndarray.cuh"

#include <tuple>

std::pair<NDArray<float3>, NDArray<int>>
dual_contouring(Grid *grid, const Intersection &its, float level = 0.0f,
                float lambda = 1e-5f, float svd_tol = 1e-6f);
