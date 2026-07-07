#pragma once

#include "grid/grid.cuh"
#include "its.cuh"
#include "ndarray.cuh"

#include <thrust/device_vector.h>

#include <tuple>

// One dual vertex per cell crossed by the surface, aligned with
// its.cell_indices: the QEF minimizer of the intersection tangent planes
// (requires normals on the intersection), or the centroid of the
// intersection points. Shared by dual contouring, surface nets and dual
// marching cubes.
thrust::device_vector<float3>
place_dual_vertices(Grid *grid, const Intersection &its, float reg = 1e-2f,
                    float svd_tol = 1e-6f, bool clamp = true);

thrust::device_vector<float3> place_centroid_vertices(const Intersection &its);

std::pair<NDArray<float3>, NDArray<int>>
dual_contouring(Grid *grid, const Intersection &its, float level = 0.0f,
                float reg = 1e-2f, float svd_tol = 1e-6f, bool clamp = true);

std::pair<NDArray<float3>, NDArray<int>>
surface_nets(Grid *grid, const Intersection &its, float level = 0.0f);
