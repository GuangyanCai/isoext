#include "mc/mc.cuh"
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

#include <array>
#include <nanobind/ndarray.h>

// Input types
using GridType = nanobind::ndarray<nanobind::pytorch, float, nanobind::ndim<3>,
                                   nanobind::device::cuda, nanobind::c_contig>;
using CellType =
    nanobind::ndarray<nanobind::pytorch, float, nanobind::shape<-1, -1, -1, 3>,
                      nanobind::device::cuda, nanobind::c_contig>;
using AABBType = std::array<float, 6>;

// Output types
using VerticesType =
    nanobind::ndarray<nanobind::pytorch, float, nanobind::shape<-1, 3>,
                      nanobind::device::cuda, nanobind::c_contig>;
using FacesType =
    nanobind::ndarray<nanobind::pytorch, int, nanobind::shape<-1, 3>,
                      nanobind::device::cuda, nanobind::c_contig>;

NB_MODULE(isoext_ext, m) {

    m.def(
        "marching_cubes",
        [](GridType grid, std::optional<AABBType> aabb,
           std::optional<CellType> cells, float level = 0.f, bool tight = true,
           std::string method = "nagae") {
            float *grid_ptr = grid.data();
            uint3 res = make_uint3(grid.shape(0), grid.shape(1), grid.shape(2));

            if (!aabb.has_value() && !cells.has_value()) {
                throw std::runtime_error(
                    "Either AABB or cell positions must be provided.");
            }
            if (aabb.has_value() && cells.has_value()) {
                throw std::runtime_error("Either AABB or cell positions must "
                                         "be provided, not both.");
            }

            thrust::device_vector<float3> cells_dv;
            float3 *cells_ptr = nullptr;

            if (aabb.has_value()) {
                auto a = aabb.value();
                cells_dv = get_cells_from_aabb(a, res);
                cells_ptr = thrust::raw_pointer_cast(cells_dv.data());
                tight = true;
            }

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

            auto [v_ptr_raw, v_len, f_ptr_raw, f_len] = mc::marching_cubes(
                grid.data(), cells_ptr, res, level, tight, method);

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
        "level"_a = 0.f, "tight"_a = true, "method"_a = "nagae",
        "Marching Cubes");

    m.doc() = "A library for extracting iso-surfaces from level-set functions";
}
