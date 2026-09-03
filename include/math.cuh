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

inline __host__ __device__ uint3
operator+(uint3 a, uint3 b) {
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
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

// Closest point to p on the triangle abc (Ericson, Real-Time Collision
// Detection, 5.1.5), by Voronoi region of the triangle's features.
inline __host__ __device__ float3
closest_point_triangle(float3 p, float3 a, float3 b, float3 c) {
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        return a;
    }
    float3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        return b;
    }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        return a + ab * (d1 / (d1 - d3));
    }
    float3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        return c;
    }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        return a + ac * (d2 / (d2 - d6));
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f) {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    float sum = va + vb + vc;
    if (sum == 0.0f) {   // degenerate triangle
        return a;
    }
    return a + ab * (vb / sum) + ac * (vc / sum);
}

// Barycentric coordinates of a point in the plane of the triangle abc;
// (1, 0, 0) when the triangle is degenerate.
inline __host__ __device__ float3
barycentric(float3 p, float3 a, float3 b, float3 c) {
    float3 v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1);
    float d20 = dot(v2, v0), d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) < 1e-12f) {
        return make_float3(1.0f, 0.0f, 0.0f);
    }
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    return make_float3(1.0f - v - w, v, w);
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