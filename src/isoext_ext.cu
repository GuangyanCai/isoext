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
    NB_TRAMPOLINE(Grid, 6);

    uint get_num_cells() const override { NB_OVERRIDE_PURE(get_num_cells); }
    uint get_num_points() const override { NB_OVERRIDE_PURE(get_num_points); }
    NDArray<float3> get_points() const override {
        NB_OVERRIDE_PURE(get_points);
    }
    NDArray<float> get_values() const override { NB_OVERRIDE_PURE(get_values); }
    void set_values(const NDArray<float> &new_values) override {
        NB_OVERRIDE_PURE(set_values, new_values);
    }
    NDArray<uint> get_cells() const override { NB_OVERRIDE_PURE(get_cells); }
};

NB_MODULE(isoext_ext, m) {

    m.def("marching_cubes", [](Grid *grid, float level, std::string method) {
        auto [v, f] = mc::marching_cubes(grid, level, method);
        if (v.size() == 0 || f.size() == 0) {
            return nb::make_tuple(nb::none(), nb::none());
        }
        return nb::make_tuple(ours_to_nb(v), ours_to_nb(f));
    });

    nb::class_<Grid, PyGrid>(m, "Grid")
        .def("get_num_cells", &Grid::get_num_cells)
        .def("get_num_points", &Grid::get_num_points)
        .def("get_points", &Grid::get_points)
        .def("get_values", &Grid::get_values)
        .def("set_values", &Grid::set_values)
        .def("get_cells", &Grid::get_cells);

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
