#pragma once

#include "common.cuh"
#include "grid/grid.cuh"
#include "math.cuh"

#include <array>

class UniformGrid : public Grid {
  private:
    NDArray<float> values;
    uint3 shape;
    float3 aabb_min, aabb_max;
    uint num_cells, num_points;

  public:
    UniformGrid(uint3 shape, float3 aabb_min, float3 aabb_max);

    ~UniformGrid() = default;

    inline uint get_num_cells() const override { return num_cells; }

    inline uint get_num_points() const override { return num_points; }

    NDArray<float3> get_points() const override;

    NDArray<float> get_values() const override;

    void set_values(const NDArray<float> &new_values) override;

    NDArray<uint> get_cells() const override;
};
