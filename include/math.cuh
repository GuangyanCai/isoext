#pragma once

#include <thrust/functional.h>
#include <cuda_runtime.h>

#include <cstdint>

inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator*(float a, float3 b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ bool operator==(float3 a, float3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline __host__ __device__ bool operator!=(float3 a, float3 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}

__host__ __device__ float3 interpolate(float level, float val_0,
                                       float val_1, float3 pos_0,
                                       float3 pos_1)
{
    if (abs(level - val_0) < 0.00001)
        return (pos_0);
    if (abs(level - val_1) < 0.00001)
        return (pos_1);
    if (abs(val_0 - val_1) < 0.00001)
        return (pos_0);
    float mu = (level - val_0) / (val_1 - val_0);
    return pos_0 + mu * (pos_1 - pos_0);
}

struct is_nan_pred
{
    __host__ __device__ bool operator()(const float3 &v)
    {
        return isnan(v.x) || isnan(v.y) || isnan(v.z);
    }
};

struct float3_less_pred : public thrust::binary_function<float3, float3, bool>
{
    __host__ __device__ bool operator()(const float3 &lhs,
                                        const float3 &rhs) const
    {
        return thrust::make_tuple(lhs.x, lhs.y, lhs.z) <
               thrust::make_tuple(rhs.x, rhs.y, rhs.z);
    }
};

struct float3_elem_eq_pred
    : public thrust::binary_function<float3, float3, bool>
{
    __host__ __device__ bool operator()(const float3 &lhs,
                                        const float3 &rhs) const
    {
        return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
    }
};

struct is_empty_pred
{
    __host__ __device__ bool operator()(const uint8_t &v)
    {
        return v == 0 || v == 255;
    }
};

struct transform_aabb_functor
{
    const float3 grid_scale, aabb_min;

    transform_aabb_functor(float3 aabb_min, float3 aabb_max, float3 old_scale)
        : grid_scale((aabb_max - aabb_min) / old_scale), aabb_min(aabb_min) {}

    __host__ __device__ float3 operator()(const float3 &v) const
    {
        return v * grid_scale + aabb_min;
    }
};