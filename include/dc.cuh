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

// Shared tail of the dual methods: one quad around every crossed edge from
// the per-cell vertices (aligned with its.cell_indices), split into
// triangles and welded into an indexed mesh.
std::pair<NDArray<float3>, NDArray<int>>
build_dual_mesh(Grid *grid, const Intersection &its,
                const thrust::device_vector<float3> &dual_v);

// Dual contouring of Hermite data (Ju et al. 2002): the QEF of the
// intersection tangent planes.
std::pair<NDArray<float3>, NDArray<int>>
dual_contouring(Grid *grid, const Intersection &its, float level = 0.0f,
                float reg = 1e-2f, float svd_tol = 1e-6f, bool clamp = true);

// Dual contouring of signed distance data (Carrera et al. 2026): vertices
// optimized against the SDF samples themselves, without normals. Defaults
// follow the authors' reference code.
struct SdfDcOptions {
    int outer_iters = 100;
    int inner_iters = 100;
    float mu = 0.1f;                // step regularization of the inner loop
    float hermite_weight = 0.02f;   // w_H, weight of the Hermite plane rows
    float update_weight = 0.2f;   // w_u, blend of face points and Hermite data
    bool hermite_update = true;   // refine Hermite data from the mesh
    bool qef_assignment = true;   // first sample assignment on the QEF mesh
    float band = 3.0f;            // samples used, in cell diagonals
    float tol = 1e-5f;            // inner-loop stopping step, in cell diagonals
};

std::pair<NDArray<float3>, NDArray<int>>
dual_contouring_sdf(Grid *grid, const Intersection &its, float level,
                    const SdfDcOptions &opt);

std::pair<NDArray<float3>, NDArray<int>>
surface_nets(Grid *grid, const Intersection &its, float level = 0.0f);
