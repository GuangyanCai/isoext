#include "mc.cuh"
#include "ndarray_types.cuh"
#include "utils.cuh"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
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
        [](GridType grid, std::optional<AABBType> aabb,
           std::optional<CellType> cells, float level = 0.f,
           std::string method = "nagae") {
            float *grid_ptr = grid.data();
            std::optional<float3 *> cells_ptr;
            if (cells.has_value()) {
                auto c = cells.value();
                cells_ptr = reinterpret_cast<float3 *>(c.data());
                for (int i = 0; i < 3; i++) {
                    if (grid.shape(i) != c.shape(i)) {
                        throw std::runtime_error(
                            "Resolutions of grid and cells must match except "
                            "for the last dimension of cells.");
                    }
                }
            }

            uint3 res = make_uint3(grid.shape(0), grid.shape(1), grid.shape(2));
            auto [v_ptr_raw, v_len, f_ptr_raw, f_len] = mc::marching_cubes(
                grid.data(), res, aabb, cells_ptr, level, method);

            if (v_len == 0 || f_len == 0) {
                cudaFree(v_ptr_raw);
                cudaFree(f_ptr_raw);
                return nb::make_tuple(nb::none(), nb::none());
            }

            VerticesType v(v_ptr_raw, {v_len, 3},
                           create_device_capsule(v_ptr_raw));
            FacesType f(f_ptr_raw, {f_len, 3},
                        create_device_capsule(f_ptr_raw));

            return nb::make_tuple(v, f);
        },
        "grid"_a, "aabb"_a = nb::none(), "cells"_a = nb::none(),
        "level"_a = 0.f, "method"_a = "nagae", "Marching Cubes");

    m.doc() = "A library for extracting iso-surfaces from level-set functions";
}
