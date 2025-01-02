#include "math.cuh"
#include "utils.cuh"

#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void
vertex_welding(thrust::device_vector<float3> &v, thrust::device_vector<int> &f,
               bool skip_scatter) {

    thrust::device_vector<float3> sorted_v;

    if (skip_scatter) {
        sorted_v = v;
    } else {
        // Scatter v to sorted_v based on f
        thrust::scatter(v.begin(), v.end(), f.begin(), sorted_v.begin());
        f.clear();
        f.resize(v.size());
        thrust::sequence(f.begin(), f.end());
    }

    // Remove duplicated vertices
    thrust::sort(sorted_v.begin(), sorted_v.end(), float3_less_pred());
    sorted_v.erase(
        thrust::unique(sorted_v.begin(), sorted_v.end(), float3_elem_eq_pred()),
        sorted_v.end());

    thrust::lower_bound(sorted_v.begin(), sorted_v.end(), v.begin(), v.end(),
                        f.begin(), float3_less_pred());

    // Update vertex array
    v = std::move(sorted_v);
}
