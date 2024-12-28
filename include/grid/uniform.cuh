#pragma once

#include "common.cuh"
#include "grid/grid.cuh"
#include "math.cuh"

#include <array>

class UniformGrid : public Grid {
  private:
    NDArray<float> values;
    NDArray<float3> points;

  public:
    UniformGrid(uint3 shape, float3 aabb_min, float3 aabb_max);

    ~UniformGrid() = default;

    NDArray<float3> get_points() const override;

    NDArray<float> get_values() const override;

    void set_values(const NDArray<float> &new_values) override;
};
