#pragma once

#include "grid/grid.cuh"
#include "math.cuh"
#include "ndarray.cuh"

#include <thrust/device_vector.h>

#include <array>

class SparseGrid : public Grid {
  private:
    thrust::device_vector<float> values;
    thrust::device_vector<uint> cell_idx;
    uint3 shape;
    float3 aabb_min, aabb_max;
    float default_value;

  public:
    SparseGrid(uint3 shape, float3 aabb_min, float3 aabb_max,
               float default_value = FMAX);

    ~SparseGrid() = default;

    inline uint get_num_cells() const override { return cell_idx.size(); }

    inline uint get_num_points() const override { return cell_idx.size() * 8; }

    NDArray<float3> get_points() const override;

    NDArray<float> get_values() const override;

    void set_values(const NDArray<float> &new_values) override;

    NDArray<uint> get_cells() const override;

    void add_cells(const NDArray<uint> &new_cell_idx);

    void remove_cells(const NDArray<uint> &new_cell_idx);

    NDArray<uint> get_cell_indices() const;
};
