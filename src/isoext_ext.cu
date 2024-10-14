#include "mc.cuh"
#include "ndarray_types.cuh"
#include "utils.cuh"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

// Function to create a nanobind capsule for device memory
nanobind::capsule
create_device_capsule(void *ptr) {
    return nanobind::capsule(ptr, [](void *p) noexcept { cudaFree(p); });
}

NB_MODULE(isoext_ext, m) {

    m.def(
        "marching_cubes",
        [](GridType grid, AABBType aabb, float level = 0.f, bool tight = true,
           std::string method = "lorensen") {
            float *grid_ptr = grid.data();
            uint3 res = make_uint3(grid.shape(0), grid.shape(1), grid.shape(2));
            float3 aabb_min = make_float3(aabb[0], aabb[1], aabb[2]);
            float3 aabb_max = make_float3(aabb[3], aabb[4], aabb[5]);

            auto [v_ptr_raw, v_len, f_ptr_raw, f_len] =
                mc::lorensen::marching_cubes(grid.data(), res, aabb_min,
                                             aabb_max, level, tight);

            VerticesType v(v_ptr_raw, {v_len, 3},
                           create_device_capsule(v_ptr_raw));
            FacesType f(f_ptr_raw, {f_len, 3},
                        create_device_capsule(f_ptr_raw));

            return nb::make_tuple(v, f);
        },
        "grid"_a, "aabb"_a, "level"_a = 0.f, "tight"_a = true,
        "method"_a = "lorensen", "Marching Cubes");

    m.doc() = "A library for extracting iso-surfaces from level-set functions";
}
