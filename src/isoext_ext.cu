#include "common.cuh"
#include "grid/uniform.cuh"
#include "math.cuh"
#include "mc/mc.cuh"
#include "utils.cuh"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/trampoline.h>

namespace nb = nanobind;
using namespace nb::literals;

template <typename DTYPE, typename... Ts>
using PyTorchCudaType =
    nb::ndarray<nb::pytorch, DTYPE, nb::device::cuda, nb::c_contig, Ts...>;

// Input types
using GridType = PyTorchCudaType<float, nb::ndim<3>>;
using CellType = PyTorchCudaType<float, nb::shape<-1, -1, -1, 3>>;
using AABBType = std::array<float, 6>;

// Output types
using VerticesType = PyTorchCudaType<float, nb::shape<-1, 3>>;
using FacesType = PyTorchCudaType<int, nb::shape<-1, 3>>;

// Function to create a nanobind capsule for device memory
nb::capsule
create_device_capsule(void *ptr) {
    return nb::capsule(ptr, [](void *p) noexcept { cudaFree(p); });
}

template <typename DTYPE, typename... Ts>
NDArray<DTYPE>
nb_to_ours(const PyTorchCudaType<DTYPE, Ts...> &arr) {
    return NDArray<DTYPE>(arr.data(),
                          {arr.shape_ptr(), arr.shape_ptr() + arr.ndim()});
}

template <typename DTYPE, typename... Ts>
PyTorchCudaType<DTYPE, Ts...>
ours_to_nb(NDArray<DTYPE> &arr) {
    NDArray<DTYPE> new_arr =
        arr.read_only ? arr : std::move(arr);   // Ensure new_arr owns the data
    new_arr.read_only = true;   // Transfer ownership to the capsule
    DTYPE *data_ptr = new_arr.data_ptr.get();
    return PyTorchCudaType<DTYPE, Ts...>(data_ptr, new_arr.ndim(),
                                         new_arr.shape.data(),
                                         create_device_capsule(data_ptr));
}

template <typename... Ts>
PyTorchCudaType<float, Ts...>
ours_to_nb(NDArray<float3> &arr) {
    float *data_ptr = reinterpret_cast<float *>(arr.data_ptr.get());
    std::vector<size_t> shape = arr.shape;
    shape.push_back(3);
    NDArray<float> new_arr(data_ptr, shape, arr.read_only);
    arr.read_only = true;
    return ours_to_nb(new_arr);
}

struct PyGrid : Grid {
    NB_TRAMPOLINE(Grid, 3);

    NDArray<float3> get_points() const override {
        NB_OVERRIDE_PURE(get_points);
    }
    NDArray<float> get_values() const override { NB_OVERRIDE_PURE(get_values); }
    void set_values(const NDArray<float> &new_values) override {
        NB_OVERRIDE_PURE(set_values, new_values);
    }
};

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

    nb::class_<Grid, PyGrid>(m, "Grid")
        .def("get_points", &Grid::get_points)
        .def("get_values", &Grid::get_values)
        .def("set_values", &Grid::set_values);

    nb::class_<UniformGrid, Grid>(m, "UniformGrid")
        .def(
            "__init__",
            [](UniformGrid *self, std::array<uint, 3> shape,
               std::array<float, 3> aabb_min, std::array<float, 3> aabb_max) {
                new (self) UniformGrid(make_uint3(shape), make_float3(aabb_min),
                                       make_float3(aabb_max));
            },
            "shape"_a, "aabb_min"_a = std::array<float, 3>{-1, -1, -1},
            "aabb_max"_a = std::array<float, 3>{1, 1, 1})
        .def("get_points",
             [](UniformGrid &self) {
                 NDArray<float3> points = self.get_points();
                 return ours_to_nb(points);
             })
        .def("get_values",
             [](UniformGrid &self) {
                 NDArray<float> values = self.get_values();
                 return ours_to_nb(values);
             })
        .def("set_values", [](UniformGrid &self, GridType new_values) {
            NDArray<float> values = nb_to_ours(new_values);
            self.set_values(values);
        });

    m.doc() = "A library for extracting iso-surfaces from level-set functions";
}
