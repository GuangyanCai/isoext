#pragma once

#include <cuda_runtime.h>
#include <thrust/functional.h>

inline __host__ __device__ float3
operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3
operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ uint3
operator-(uint3 a, uint3 b) {
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3
operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3
operator*(float a, float3 b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ float3
operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3
operator/(float3 a, float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ bool
operator==(float3 a, float3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline __host__ __device__ bool
operator!=(float3 a, float3 b) {
    return a.x != b.x || a.y != b.y || a.z != b.z;
}

template <typename T>
__host__ __device__ T
lerp(float t, T a, T b) {
    return (1 - t) * a + t * b;
}

struct is_nan_pred {
    __host__ __device__ bool operator()(const float3 &v) {
        return isnan(v.x) || isnan(v.y) || isnan(v.z);
    }
};

struct float3_less_pred {
    __host__ __device__ bool operator()(const float3 &lhs,
                                        const float3 &rhs) const {
        return thrust::make_tuple(lhs.x, lhs.y, lhs.z) <
               thrust::make_tuple(rhs.x, rhs.y, rhs.z);
    }
};

struct float3_elem_eq_pred {
    __host__ __device__ bool operator()(const float3 &lhs,
                                        const float3 &rhs) const {
        return thrust::make_tuple(lhs.x, lhs.y, lhs.z) ==
               thrust::make_tuple(rhs.x, rhs.y, rhs.z);
    }
};

struct is_empty_pred {
    __host__ __device__ bool operator()(const uint8_t &v) {
        return v == 0 || v == 255;
    }
};