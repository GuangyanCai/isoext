#include "mc.cuh"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include <thrust/device_malloc.h>

#include <cstdint>
#include <algorithm>

namespace nb = nanobind;
using namespace nb::literals;

// Input types
using GridType = nb::ndarray<float, nb::ndim<3>, nb::device::cuda, nb::c_contig>;
using AABBType = std::array<float, 6>;

// Output types
using Vertices = nb::ndarray<nb::pytorch, float, nb::shape<-1, 3>, nb::device::cuda, nb::c_contig>;
using Faces = nb::ndarray<nb::pytorch, int, nb::shape<-1, 3>, nb::device::cuda, nb::c_contig>;

NB_MODULE(isoext_ext, m) {

    m.def("marching_cubes", [](GridType grid, AABBType aabb, float level = 0.f, std::string method = "lorensen"){
        // Convert grid shape to a std::array.
        std::array<int64_t, 3> grid_shape;
        std::copy_n(grid.shape_ptr(), 3, grid_shape.begin());

        // Run marching_cubes to get the vertices and faces.
        float * v_ptr_raw;
        int * f_ptr_raw;
        uint32_t v_len, f_len;

        if (method == "lorensen") {
            std::tie(v_ptr_raw, v_len, f_ptr_raw, f_len) = mc::lorensen::marching_cubes(grid.data(), grid_shape, aabb, level);
        }
        else {
            throw std::invalid_argument("Invalid method.");
        }

        nb::capsule v_owner(v_ptr_raw, [](void *p) noexcept {
             cudaFree(p);
        });

        nb::capsule f_owner(f_ptr_raw, [](void *p) noexcept {
             cudaFree(p);
        });

        // Convert the pointers into nb::ndarray.
        auto v_ndarray = Vertices(v_ptr_raw, {v_len, 3}, v_owner);
        auto f_ndarray = Faces(f_ptr_raw, {f_len, 3}, f_owner);

        // Return a tuple of (v, f).
        return nb::make_tuple(v_ndarray, f_ndarray);
    }, "grid"_a, "aabb"_a, "level"_a = 0.f, "method"_a = "lorensen", "Marching Cubes");

    m.doc() = "A library for extracting iso-surfaces from level-set functions";
    
}
