#pragma once

#include "grid/grid.cuh"
#include "ndarray.cuh"

struct Intersection {
    NDArray<float3> points;
    NDArray<float3> normals;
    NDArray<uint> cell_offsets;

    Intersection(uint num_points, NDArray<uint> &&cell_offsets)
        : points({num_points}), normals({num_points}),
          cell_offsets(std::move(cell_offsets)) {}

    inline NDArray<float3> get_points() { return points; }
    inline NDArray<float3> get_normals() { return normals; }
    inline void set_normals(const NDArray<float3> &normals_) {
        normals.set(normals_);
    }
};

Intersection get_intersection(Grid *grid, float level);