#include "grid/uniform.cuh"
#include "utils.cuh"

UniformGrid::UniformGrid(uint3 shape, float3 aabb_min, float3 aabb_max,
                         float default_value)
    : shape(shape), aabb_min(aabb_min), aabb_max(aabb_max),
      num_cells((shape.x - 1) * (shape.y - 1) * (shape.z - 1)),
      num_points(shape.x * shape.y * shape.z) {
    size_t size_check = (size_t) shape.x * (size_t) shape.y * (size_t) shape.z;
    if (size_check > INT_MAX) {
        throw std::runtime_error(
            "Number of points exceeds maximum value 2147483647 (max int)");
    }
    values = NDArray<float>({shape.x, shape.y, shape.z});
    values.fill(default_value);
}

NDArray<float3>
UniformGrid::get_points() const {
    NDArray<float3> points({shape.x, shape.y, shape.z});
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
