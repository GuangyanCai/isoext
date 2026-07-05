#pragma once

#include "grid/grid.cuh"
#include "ndarray.cuh"

#include <utility>

// Marching tetrahedra: split every cell into 6 tetrahedra and extract the
// surface per tetrahedron. A tetrahedron has no ambiguous sign
// configurations, so the mesh is watertight and consistent by
// construction, at the cost of roughly 2-3x more triangles than marching
// cubes. After Doi and Koide, "An efficient method of triangulating
// equi-valued surfaces by using tetrahedral cells" (1991).
std::pair<NDArray<float3>, NDArray<int>>
marching_tetrahedra(Grid *grid, float level = 0.0f);
