#pragma once

#include "lut.cuh"
#include "math.cuh"
#include "utils.cuh"

namespace mc {
namespace lorensen {

void run(const thrust::device_vector<uint8_t> &case_idx_dv,
         const thrust::device_vector<uint32_t> &grid_idx_dv, float3 *v,
         const float *grid, const float3 *cells, const uint3 res, float level,
         bool tight);

}   // namespace lorensen
}   // namespace mc