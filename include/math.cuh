#pragma once

#include <cuda_runtime.h>
#include <thrust/functional.h>

#include <array>
#include <limits>

using uint = unsigned int;
constexpr float FMAX = std::numeric_limits<float>::max();

inline float3
make_float3(std::array<float, 3> xyz) {
    return make_float3(xyz[0], xyz[1], xyz[2]);
}

inline uint3
make_uint3(std::array<uint, 3> xyz) {
    return make_uint3(xyz[0], xyz[1], xyz[2]);
}

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

inline __host__ __device__ float3
operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ uint3
operator-(uint3 a, uint b) {
    return make_uint3(a.x - b, a.y - b, a.z - b);
}

template <typename T>
__host__ __device__ T
lerp(float t, T a, T b) {
    return (1 - t) * a + t * b;
}

inline __host__ __device__ float
dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3
cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

inline __host__ __device__ float
norm(float3 a) {
    return sqrt(dot(a, a));
}

inline __host__ __device__ float3
clip(float3 a, float3 min, float3 max) {
    return make_float3(fminf(fmaxf(a.x, min.x), max.x),
                       fminf(fmaxf(a.y, min.y), max.y),
                       fminf(fmaxf(a.z, min.z), max.z));
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

struct uint2_less_pred {
    __host__ __device__ bool operator()(const uint2 &lhs,
                                        const uint2 &rhs) const {
        return thrust::make_tuple(lhs.x, lhs.y) <
               thrust::make_tuple(rhs.x, rhs.y);
    }
};

struct uint2_equal_pred {
    __host__ __device__ bool operator()(const uint2 &lhs,
                                        const uint2 &rhs) const {
        return thrust::make_tuple(lhs.x, lhs.y) ==
               thrust::make_tuple(rhs.x, rhs.y);
    }
};

struct is_empty_pred {
    __host__ __device__ bool operator()(const uint8_t &v) {
        return v == 0 || v == 255;
    }
};

struct is_zero_pred {
    __host__ __device__ bool operator()(const int &v) { return v == 0; }
};