#pragma once

#include "lorensen.cuh"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

template <typename Framework, typename Device>
struct MarchingCubes {
    // Input types
    using GridType = nb::ndarray<Framework, float, nb::ndim<3>, Device, nb::c_contig>;
    using AABBType = std::array<float, 6>;

    // Output types
    using Vertices = nb::ndarray<Framework, float, nb::shape<-1, 3>, Device, nb::c_contig>;
    using Faces = nb::ndarray<Framework, int, nb::shape<-1, 3>, Device, nb::c_contig>;

    static nb::tuple run(GridType grid, AABBType aabb, float level = 0.f, std::string method = "lorensen") {
        bool from_cpu = grid.device_type() == nb::device::cpu::value;
        
        // Convert grid shape to a std::array.
        std::array<int64_t, 3> grid_shape;
        std::copy_n(grid.shape_ptr(), 3, grid_shape.begin());

        // Run marching_cubes to get the vertices and faces.
        float * v_ptr_raw;
        int * f_ptr_raw;
        uint32_t v_len, f_len;

        float * grid_data = grid.data();
        if (from_cpu) {
            grid_data = host_to_device(grid.data(), grid.size());
        }
        
        if (method == "lorensen") {
            std::tie(v_ptr_raw, v_len, f_ptr_raw, f_len) = mc::lorensen::marching_cubes(grid_data, grid_shape, aabb, level);
        }
        else {
            throw std::invalid_argument("Invalid method.");
        }

        nb::capsule v_owner, f_owner;

        if (from_cpu) {
            float * tmp_v_ptr_raw = v_ptr_raw;
            int * tmp_f_ptr_raw = f_ptr_raw;
            v_ptr_raw = device_to_host(tmp_v_ptr_raw, v_len * 3);
            f_ptr_raw = device_to_host(tmp_f_ptr_raw, f_len * 3);
            cudaFree(tmp_v_ptr_raw);
            cudaFree(tmp_f_ptr_raw);

            v_owner = create_host_capsule(v_ptr_raw);
            f_owner = create_host_capsule(f_ptr_raw);
        }
        else {
            v_owner = create_device_capsule(v_ptr_raw);
            f_owner = create_device_capsule(f_ptr_raw);
        }

        // Convert the pointers into nb::ndarray.
        Vertices v_ndarray(v_ptr_raw, {v_len, 3}, v_owner);
        Faces f_ndarray(f_ptr_raw, {f_len, 3}, f_owner);

        // Return a tuple of (v, f).
        return nb::make_tuple(v_ndarray, f_ndarray);
    }
};