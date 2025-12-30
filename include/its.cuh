#pragma once

#include "grid/grid.cuh"
#include "ndarray.cuh"

struct Intersection {
    NDArray<float3> points;
    NDArray<float3> normals;
    NDArray<uint2> edges;
    NDArray<uint> cell_indices;
    NDArray<uint> cell_offsets;
    NDArray<bool> is_out;
    bool _has_normals;

    Intersection(uint num_points)
        : points({num_points}), normals({num_points}), edges({num_points}),
          is_out({num_points}), _has_normals(false) {}

    inline NDArray<float3> get_points() { return points; }
    inline NDArray<float3> get_normals() { return normals; }
    inline bool has_normals() const { return _has_normals; }
    inline void set_normals(const NDArray<float3> &normals_) {
        normals.set(normals_);
        _has_normals = true;
    }
};

Intersection get_intersection(Grid *grid, float level, bool compute_normals = false);

// Compute normals at intersection points using central differences on grid values
void compute_intersection_normals(Intersection &its, Grid *grid);