#pragma once

#include "math.cuh"
#include "ndarray.cuh"

#include <memory>
#include <tuple>

// A triangle mesh with a GPU bounding volume hierarchy over it, for
// distance queries: the closest point on the mesh to each query point, and
// whether a query point lies inside the (closed) mesh. Built once, queried
// many times. The BVH build and traversal come from cuBQL (ext/cuBQL).
class MeshBVH {
  public:
    // vertices: (V,) float3 positions; faces: (F, 3) vertex indices.
    MeshBVH(const NDArray<float3> &vertices, const NDArray<int> &faces);
    ~MeshBVH();

    uint num_faces() const;

    // For each point: the distance to the mesh, the closest point on it and
    // the index of the triangle holding that point.
    std::tuple<NDArray<float>, NDArray<float3>, NDArray<int>>
    closest(const NDArray<float3> &points) const;

    // +1 outside, -1 inside, decided by a majority vote of six rays'
    // crossing parities. Meaningful for closed meshes only.
    NDArray<float> sign(const NDArray<float3> &points) const;

    // Generalized winding number of each point: 1 inside a closed mesh, 0
    // outside, and a graceful in-between for meshes with holes or soups.
    // Evaluated exactly near the point and by a second-order expansion of
    // each far BVH node (Barill et al. 2018).
    NDArray<float> winding_number(const NDArray<float3> &points) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
