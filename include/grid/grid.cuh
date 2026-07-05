#pragma once

#include "math.cuh"
#include "ndarray.cuh"
#include "utils.cuh"

#include <thrust/device_vector.h>

#include <stdexcept>
#include <tuple>

// A non-owning, device-copyable view of a grid used by the extraction
// kernels. Corner indices, positions and values are computed on the fly
// instead of materializing the full cell and point arrays, which keeps the
// kernels memory-bound work proportional to the number of cells processed.
//
// A dense grid (UniformGrid) derives everything from the cell index alone.
// A sparse grid additionally looks up the dense index of each active cell
// and stores its values per cell corner instead of per lattice point.
//
// The pointers borrow the grid's storage: a view must not outlive its grid.
struct GridView {
    uint3 shape;   // point lattice resolution
    float3 aabb_min, aabb_max;
    const float *values;        // dense: per lattice point; sparse: per corner
    const uint *cell_indices;   // sparse only: active cell -> dense cell index
    bool sparse;

    // Index of a cell corner in the full point lattice. Corner i has local
    // coordinates (x, y, z) = (bit 2, bit 1, bit 0) of i (Morton order).
    __host__ __device__ uint dense_point_index(uint cell, uint corner) const {
        uint dense_cell = sparse ? cell_indices[cell] : cell;
        uint3 cell_3d = idx_1d_to_3d(dense_cell, shape - 1);
        uint3 offset =
            make_uint3((corner >> 2) & 1, (corner >> 1) & 1, corner & 1);
        return idx_3d_to_1d(cell_3d + offset, shape);
    }

    // Identifier of a cell corner as stored in Intersection::edges. For a
    // sparse grid this indexes the per-corner value array; for a dense grid
    // it is the lattice point index.
    __host__ __device__ uint corner_point_id(uint cell, uint corner) const {
        return sparse ? cell * 8 + corner : dense_point_index(cell, corner);
    }

    __host__ __device__ float corner_value(uint cell, uint corner) const {
        return values[corner_point_id(cell, corner)];
    }

    __host__ __device__ float3 corner_position(uint cell, uint corner) const {
        return get_vtx_pos_op(shape, aabb_min,
                              aabb_max)(dense_point_index(cell, corner));
    }

    // Load the positions and values of all 8 corners of a cell.
    __host__ __device__ void load_corners(uint cell, float3 p[8],
                                          float v[8]) const {
        for (uint i = 0; i < 8; i++) {
            p[i] = corner_position(cell, i);
            v[i] = corner_value(cell, i);
        }
    }
};

// Compute the marching cubes case number of each cell.
struct get_case_num_op {
    uint8_t *cases;
    const GridView view;
    const float level;

    get_case_num_op(uint8_t *cases, const GridView &view, const float level)
        : cases(cases), view(view), level(level) {}

    __host__ __device__ void operator()(uint cell_idx) {
        // Compute the sign of each cube vertex and derive the case number
        uint8_t case_num = 0;
        for (uint i = 0; i < 8; i++) {
            case_num |= (view.corner_value(cell_idx, i) - level < 0) << i;
        }
        cases[cell_idx] = case_num;
    }
};

class Grid {
  public:
    Grid() = default;

    virtual ~Grid() = default;

    virtual uint get_num_cells() const = 0;

    virtual uint get_num_points() const = 0;

    virtual uint3 get_shape() const = 0;

    // Device-side view consumed by the extraction algorithms. Not pure so
    // that Python-defined Grid subclasses remain constructible; C++ grids
    // must override it.
    virtual GridView get_view() const {
        throw std::runtime_error(
            "get_view is not implemented for this grid type");
    }

    virtual NDArray<float3> get_points() const = 0;

    virtual NDArray<float> get_values() const = 0;

    virtual void set_values(const NDArray<float> &new_values) = 0;

    virtual NDArray<uint> get_cells() const = 0;

    virtual thrust::device_vector<uint> get_cell_indices() const = 0;

    virtual std::tuple<thrust::device_vector<int4>, thrust::device_vector<bool>>
    get_dual_quads(const NDArray<uint2> &edges,
                   const NDArray<bool> &is_out) const = 0;
};
