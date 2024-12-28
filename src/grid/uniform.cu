#include "grid/uniform.cuh"
#include "utils.cuh"

namespace {
struct idx_to_cell_op {
    uint *cells;
    uint3 shape;

    __host__ __device__ idx_to_cell_op(uint *cells, uint3 shape)
        : cells(cells), shape(shape) {}

    __host__ __device__ void operator()(uint idx) {
        uint x, y, z, yz, i;
        i = idx;
        z = i % (shape.z - 1);
        i /= (shape.z - 1);
        y = i % (shape.y - 1);
        x = i / (shape.y - 1);
        idx *= 8;
        yz = shape.y * shape.z;
        cells[idx + 0] = x * yz + y * shape.z + z;
        cells[idx + 1] = cells[idx + 0] + 1;
        cells[idx + 2] = cells[idx + 0] + shape.z;
        cells[idx + 3] = cells[idx + 1] + shape.z;
        cells[idx + 4] = cells[idx + 0] + yz;
        cells[idx + 5] = cells[idx + 1] + yz;
        cells[idx + 6] = cells[idx + 2] + yz;
        cells[idx + 7] = cells[idx + 3] + yz;
    }
};
}   // anonymous namespace

UniformGrid::UniformGrid(uint3 shape, float3 aabb_min, float3 aabb_max)
    : values(shape), shape(shape), aabb_min(aabb_min), aabb_max(aabb_max),
      num_cells((shape.x - 1) * (shape.y - 1) * (shape.z - 1)),
      num_points(shape.x * shape.y * shape.z) {}

NDArray<float3>
UniformGrid::get_points() const {
    NDArray<float3> points(shape);
    thrust::transform(thrust::counting_iterator<uint>(0),
                      thrust::counting_iterator<uint>(num_points),
                      points.data_ptr,
                      get_vtx_pos_op(shape, aabb_min, aabb_max));
    return points;
}

NDArray<float>
UniformGrid::get_values() const {
    return values;
}

void
UniformGrid::set_values(const NDArray<float> &new_values) {
    values.set(new_values);
}

NDArray<uint>
UniformGrid::get_cells() const {
    NDArray<uint> cells({shape.x - 1, shape.y - 1, shape.z - 1, 8});
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_cells),
                     idx_to_cell_op(cells.data_ptr.get(), shape));
    return cells;
}
