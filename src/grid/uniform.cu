#include "grid/uniform.cuh"
#include "utils.cuh"

UniformGrid::UniformGrid(uint3 shape, float3 aabb_min, float3 aabb_max)
    : values({shape.x, shape.y, shape.z}), points({shape.x, shape.y, shape.z}) {
    uint num_points = shape.x * shape.y * shape.z;

    thrust::transform(thrust::counting_iterator<uint>(0),
                      thrust::counting_iterator<uint>(num_points),
                      points.data_ptr,
                      get_vtx_pos_op(shape, aabb_min, aabb_max));
}

NDArray<float3>
UniformGrid::get_points() const {
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
