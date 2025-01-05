#pragma once

#include "math.cuh"
#include "ndarray.cuh"

#include <thrust/device_vector.h>

#include <tuple>

class Grid {
  public:
    Grid() = default;

    virtual ~Grid() = default;

    virtual uint get_num_cells() const = 0;

    virtual uint get_num_points() const = 0;

    virtual uint3 get_shape() const = 0;

    virtual NDArray<float3> get_points() const = 0;

    virtual NDArray<float> get_values() const = 0;

    virtual void set_values(const NDArray<float> &new_values) = 0;

    virtual NDArray<uint> get_cells() const = 0;

    virtual thrust::device_vector<uint> get_cell_indices() const = 0;

    virtual std::tuple<thrust::device_vector<int4>, thrust::device_vector<bool>>
    get_dual_quads(const NDArray<uint2> &edges,
                   const NDArray<bool> &is_out) const = 0;
};
