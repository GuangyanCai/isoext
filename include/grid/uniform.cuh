#pragma once

#include "grid/grid.cuh"
#include "ndarray.cuh"

#include <array>

class UniformGrid : public Grid {
  private:
    NDArray<float> values;
    uint3 shape;
    float3 aabb_min, aabb_max;
    uint num_cells, num_points;

  public:
    UniformGrid(uint3 shape, float3 aabb_min, float3 aabb_max,
                float default_value = FMAX);

    ~UniformGrid() = default;

    inline uint get_num_cells() const override { return num_cells; }

    inline uint get_num_points() const override { return num_points; }

    inline uint3 get_shape() const override { return shape; }

    NDArray<float3> get_points() const override;

    NDArray<float> get_values() const override;

    void set_values(const NDArray<float> &new_values) override;

    NDArray<uint> get_cells() const override;

    thrust::device_vector<uint> get_cell_indices() const override;

    std::tuple<thrust::device_vector<int4>, thrust::device_vector<bool>>
    get_dual_quads(const NDArray<uint2> &edges,
                   const NDArray<bool> &is_out) const override;
};
