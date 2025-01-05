#pragma once

#include "grid/grid.cuh"
#include "math.cuh"
#include "ndarray.cuh"

#include <thrust/device_vector.h>

#include <array>

class SparseGrid : public Grid {
  private:
    thrust::device_vector<float> values;
    thrust::device_vector<uint> cell_indices;
    uint3 shape;
    float3 aabb_min, aabb_max;
    float default_value;

    NDArray<float3>
    get_points_from_cell_indices(const NDArray<uint> &cell_indices) const;

  public:
    SparseGrid(uint3 shape, float3 aabb_min, float3 aabb_max,
               float default_value = FMAX);

    ~SparseGrid() = default;

    inline uint get_num_cells() const override { return cell_indices.size(); }

    inline uint get_num_points() const override {
        return cell_indices.size() * 8;
    }

    inline uint3 get_shape() const override { return shape; }

    NDArray<float3> get_points() const override;

    NDArray<float> get_values() const override;

    void set_values(const NDArray<float> &new_values) override;

    NDArray<uint> get_cells() const override;

    void add_cells(const NDArray<uint> &new_cell_indices);

    void remove_cells(const NDArray<uint> &new_cell_indices);

    thrust::device_vector<uint> get_cell_indices() const override;

    std::vector<NDArray<int>> get_potential_cell_indices(uint chunk_size) const;

    NDArray<float3>
    get_points_by_cell_indices(const NDArray<uint> &new_cell_indices) const;

    NDArray<uint> filter_cell_indices(const NDArray<uint> &new_cell_indices,
                                      const NDArray<float> &new_values,
                                      float level = 0.f) const;

    std::tuple<thrust::device_vector<int4>, thrust::device_vector<bool>>
    get_dual_quads(const NDArray<uint2> &edges,
                   const NDArray<bool> &is_out) const override;
};
