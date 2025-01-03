#include "dc.cuh"
#include "grid/sparse.cuh"
#include "grid/uniform.cuh"
#include "its.cuh"
#include "mc/mc.cuh"
#include "ndarray.cuh"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/trampoline.h>

#include <limits>

namespace nb = nanobind;
using namespace nb::literals;

template <typename DTYPE, typename... Ts>
using PyTorchCuda =
    nb::ndarray<nb::pytorch, DTYPE, nb::device::cuda, nb::c_contig, Ts...>;

// Input types
using UniformGridData = PyTorchCuda<float, nb::ndim<3>>;
using SparseGridData = PyTorchCuda<float, nb::shape<-1, 8>>;
using SparseGridCellIndices = PyTorchCuda<int, nb::ndim<1>>;
using Vector3 = PyTorchCuda<float, nb::shape<-1, 3>>;

// Function to create a nanobind capsule for device memory
nb::capsule
create_device_capsule(void *ptr) {
    return nb::capsule(ptr, [](void *p) noexcept { cudaFree(p); });
}

template <typename DTYPE, typename... Ts>
NDArray<DTYPE>
nb_to_ours(const PyTorchCuda<DTYPE, Ts...> &arr) {
    return NDArray<DTYPE>(arr.data(),
                          {arr.shape_ptr(), arr.shape_ptr() + arr.ndim()});
}

template <typename DTYPE, typename... Ts>
PyTorchCuda<DTYPE, Ts...>
ours_to_nb(NDArray<DTYPE> &arr) {
    if (arr.size() == 0) {
        return PyTorchCuda<DTYPE, Ts...>();
    }
    NDArray<DTYPE> new_arr =
        arr.read_only ? arr : std::move(arr);   // Ensure new_arr owns the data
    new_arr.read_only = true;   // Transfer ownership to the capsule
    DTYPE *data_ptr = new_arr.data_ptr.get();
    return PyTorchCuda<DTYPE, Ts...>(data_ptr, new_arr.ndim(),
                                     new_arr.shape.data(),
                                     create_device_capsule(data_ptr));
}

// special case for Vector3
NDArray<float3>
nb_to_ours(const Vector3 &arr) {
    float3 *data_ptr = reinterpret_cast<float3 *>(arr.data());
    return NDArray<float3>(data_ptr, {arr.shape(0)});
}

// special case for float3
template <typename... Ts>
PyTorchCuda<float, Ts...>
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

    m.def(
        "marching_cubes",
        [](Grid *grid, float level, std::string method) {
            auto [v, f] = mc::marching_cubes(grid, level, method);
            return nb::make_tuple(ours_to_nb(v), ours_to_nb(f));
        },
        "grid"_a, "level"_a = 0.f, "method"_a = "nagae");

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
               std::array<float, 3> aabb_min, std::array<float, 3> aabb_max,
               float default_value) {
                new (self) UniformGrid(make_uint3(shape), make_float3(aabb_min),
                                       make_float3(aabb_max), default_value);
            },
            "shape"_a, "aabb_min"_a = std::array<float, 3>{-1, -1, -1},
            "aabb_max"_a = std::array<float, 3>{1, 1, 1},
            "default_value"_a = FMAX)
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
        .def("set_values", [](UniformGrid &self, UniformGridData new_values) {
            NDArray<float> values = nb_to_ours(new_values);
            self.set_values(values);
        });

    nb::class_<SparseGrid, Grid>(m, "SparseGrid")
        .def(
            "__init__",
            [](SparseGrid *self, std::array<uint, 3> shape,
               std::array<float, 3> aabb_min, std::array<float, 3> aabb_max,
               float default_value) {
                new (self) SparseGrid(make_uint3(shape), make_float3(aabb_min),
                                      make_float3(aabb_max), default_value);
            },
            "shape"_a, "aabb_min"_a = std::array<float, 3>{-1, -1, -1},
            "aabb_max"_a = std::array<float, 3>{1, 1, 1},
            "default_value"_a = std::numeric_limits<float>::max())
        .def("get_num_cells", &SparseGrid::get_num_cells)
        .def("get_num_points", &SparseGrid::get_num_points)
        .def("get_points",
             [](SparseGrid &self) {
                 NDArray<float3> points = self.get_points();
                 return ours_to_nb(points);
             })
        .def("get_values",
             [](SparseGrid &self) {
                 NDArray<float> values = self.get_values();
                 return ours_to_nb(values);
             })
        .def("set_values",
             [](SparseGrid &self, SparseGridData new_values) {
                 NDArray<float> values = nb_to_ours(new_values);
                 self.set_values(values);
             })
        .def("get_cells",
             [](SparseGrid &self) {
                 NDArray<uint> cells = self.get_cells();
                 return ours_to_nb(cells);
             })
        .def("add_cells",
             [](SparseGrid &self, SparseGridCellIndices new_cell_indices) {
                 NDArray<uint> new_cell_indices_ =
                     nb_to_ours(new_cell_indices).cast<uint>();
                 self.add_cells(new_cell_indices_);
             })
        .def("remove_cells",
             [](SparseGrid &self, SparseGridCellIndices new_cell_indices) {
                 NDArray<uint> new_cell_indices_ =
                     nb_to_ours(new_cell_indices).cast<uint>();
                 self.remove_cells(new_cell_indices_);
             })
        .def("get_cell_indices", [](SparseGrid &self) {
            NDArray<int> cell_indices = self.get_cell_indices().cast<int>();
            return ours_to_nb(cell_indices);
        });

    nb::class_<Intersection>(m, "Intersection")
        .def("get_points",
             [](Intersection &self) { return ours_to_nb(self.points); })
        .def("get_normals",
             [](Intersection &self) { return ours_to_nb(self.normals); })
        .def("set_normals", [](Intersection &self, Vector3 new_normals) {
            NDArray<float3> normals = nb_to_ours(new_normals);
            self.set_normals(normals);
        });

    m.def("get_intersection", &get_intersection, "grid"_a, "level"_a = 0.f);

    m.def(
        "dual_contouring",
        [](Grid *grid, Intersection its, float level, float lambda,
           float svd_tol) {
            auto [v, f] = dual_contouring(grid, its, level, lambda, svd_tol);
            return nb::make_tuple(ours_to_nb(v), ours_to_nb(f));
        },
        "grid"_a, "its"_a, "level"_a = 0.f, "lambda"_a = 1e-5f,
        "svd_tol"_a = 1e-6f);

    m.doc() = "A library for extracting iso-surfaces from level-set functions";
}
