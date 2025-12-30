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
#include <nanobind/stl/vector.h>
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
        "grid"_a, "level"_a = 0.f, "method"_a = "nagae",
        "Extract an iso-surface from a grid using the Marching Cubes algorithm.\n\n"
        "Args:\n"
        "    grid: The input grid (UniformGrid or SparseGrid) containing scalar values.\n"
        "    level: The iso-value at which to extract the surface. Default is 0.0.\n"
        "    method: The marching cubes variant to use. Options are 'nagae' (default) or 'lorensen'.\n\n"
        "Returns:\n"
        "    A tuple (vertices, faces) where vertices is an (N, 3) float32 tensor of vertex positions\n"
        "    and faces is an (M, 3) uint32 tensor of triangle indices.");

    nb::class_<Grid, PyGrid>(m, "Grid",
        "Abstract base class for all grid types.\n\n"
        "Grids store scalar values at discrete points and define the topology of cells.\n"
        "Use UniformGrid for dense regular grids or SparseGrid for adaptive grids.")
        .def("get_num_cells", &Grid::get_num_cells,
            "Return the number of cells in the grid.")
        .def("get_num_points", &Grid::get_num_points,
            "Return the number of points (vertices) in the grid.")
        .def("get_points", &Grid::get_points,
            "Return the 3D coordinates of all grid points as an (N, 3) float32 tensor.")
        .def("get_values", &Grid::get_values,
            "Return the scalar values at all grid points.")
        .def("set_values", &Grid::set_values,
            "Set the scalar values at all grid points.")
        .def("get_cells", &Grid::get_cells,
            "Return the cell connectivity as a tensor of point indices.");

    nb::class_<UniformGrid, Grid>(m, "UniformGrid",
        "A dense uniform grid for storing scalar values.\n\n"
        "The grid divides a 3D axis-aligned bounding box into a regular lattice of cells.\n"
        "Each cell has 8 corner points where scalar values are stored.")
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
            "default_value"_a = FMAX,
            "Create a uniform grid.\n\n"
            "Args:\n"
            "    shape: The number of cells in each dimension (x, y, z).\n"
            "    aabb_min: The minimum corner of the bounding box. Default is [-1, -1, -1].\n"
            "    aabb_max: The maximum corner of the bounding box. Default is [1, 1, 1].\n"
            "    default_value: Initial scalar value for all points. Default is float max.")
        .def("get_points",
             [](UniformGrid &self) {
                 NDArray<float3> points = self.get_points();
                 return ours_to_nb(points);
             },
             "Return the 3D coordinates of all grid points as a (X, Y, Z, 3) float32 tensor.")
        .def("get_values",
             [](UniformGrid &self) {
                 NDArray<float> values = self.get_values();
                 return ours_to_nb(values);
             },
             "Return the scalar values as a (X, Y, Z) float32 tensor.")
        .def(
            "set_values",
            [](UniformGrid &self, UniformGridData new_values) {
                NDArray<float> values = nb_to_ours(new_values);
                self.set_values(values);
            },
            "new_values"_a,
            "Set the scalar values from a (X, Y, Z) float32 tensor.");

    nb::class_<SparseGrid, Grid>(m, "SparseGrid",
        "A sparse adaptive grid for storing scalar values.\n\n"
        "Unlike UniformGrid, SparseGrid only allocates memory for cells that are explicitly added.\n"
        "This is useful for large domains where only a small region contains the iso-surface.")
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
            "default_value"_a = std::numeric_limits<float>::max(),
            "Create a sparse grid.\n\n"
            "Args:\n"
            "    shape: The maximum number of cells in each dimension (x, y, z).\n"
            "    aabb_min: The minimum corner of the bounding box. Default is [-1, -1, -1].\n"
            "    aabb_max: The maximum corner of the bounding box. Default is [1, 1, 1].\n"
            "    default_value: Default scalar value for unset points. Default is float max.")
        .def("get_num_cells", &SparseGrid::get_num_cells,
            "Return the number of active cells in the grid.")
        .def("get_num_points", &SparseGrid::get_num_points,
            "Return the number of points in active cells (num_cells * 8).")
        .def("get_points",
             [](SparseGrid &self) {
                 NDArray<float3> points = self.get_points();
                 return ours_to_nb(points);
             },
             "Return the 3D coordinates of points in active cells as an (N, 8, 3) float32 tensor.")
        .def("get_values",
             [](SparseGrid &self) {
                 NDArray<float> values = self.get_values();
                 return ours_to_nb(values);
             },
             "Return the scalar values at active cell corners as an (N, 8) float32 tensor.")
        .def(
            "set_values",
            [](SparseGrid &self, SparseGridData new_values) {
                NDArray<float> values = nb_to_ours(new_values);
                self.set_values(values);
            },
            "new_values"_a,
            "Set the scalar values from an (N, 8) float32 tensor.")
        .def("get_cells",
             [](SparseGrid &self) {
                 NDArray<uint> cells = self.get_cells();
                 return ours_to_nb(cells);
             },
             "Return the cell connectivity as point indices.")
        .def(
            "add_cells",
            [](SparseGrid &self, SparseGridCellIndices new_cell_indices) {
                NDArray<uint> new_cell_indices_ =
                    nb_to_ours(new_cell_indices).cast<uint>();
                self.add_cells(new_cell_indices_);
            },
            "new_cell_indices"_a,
            "Add cells to the grid by their linear indices.\n\n"
            "Args:\n"
            "    new_cell_indices: 1D int32 tensor of cell indices to add.")
        .def(
            "remove_cells",
            [](SparseGrid &self, SparseGridCellIndices new_cell_indices) {
                NDArray<uint> new_cell_indices_ =
                    nb_to_ours(new_cell_indices).cast<uint>();
                self.remove_cells(new_cell_indices_);
            },
            "new_cell_indices"_a,
            "Remove cells from the grid by their linear indices.\n\n"
            "Args:\n"
            "    new_cell_indices: 1D int32 tensor of cell indices to remove.")
        .def("get_cell_indices",
             [](SparseGrid &self) {
                 thrust::device_vector<uint> cell_indices_dv =
                     self.get_cell_indices();
                 NDArray<int> cell_indices =
                     NDArray<uint>::copy(cell_indices_dv.data().get(),
                                         {cell_indices_dv.size()})
                         .cast<int>();
                 return ours_to_nb(cell_indices);
             },
             "Return the linear indices of all active cells as a 1D int32 tensor.")
        .def("get_potential_cell_indices",
             [](SparseGrid &self, uint chunk_size) {
                 auto cell_indices =
                     self.get_potential_cell_indices(chunk_size);
                 std::vector<PyTorchCuda<int>> cell_indices_vec;
                 for (auto &cell_indices_ : cell_indices) {
                     cell_indices_vec.push_back(ours_to_nb(cell_indices_));
                 }
                 return cell_indices_vec;
             },
             "Get potential cell indices in chunks for memory-efficient processing.\n\n"
             "Args:\n"
             "    chunk_size: Maximum number of cells per chunk.\n\n"
             "Returns:\n"
             "    A list of 1D int32 tensors, each containing cell indices for a chunk.")
        .def(
            "get_points_by_cell_indices",
            [](SparseGrid &self, SparseGridCellIndices cell_indices) {
                NDArray<int> cell_indices_int = nb_to_ours(cell_indices);
                NDArray<uint> cell_indices_uint = cell_indices_int.cast<uint>();
                NDArray<float3> points =
                    self.get_points_by_cell_indices(cell_indices_uint);
                return ours_to_nb(points);
            },
            "cell_indices"_a,
            "Get the corner points of specified cells.\n\n"
            "Args:\n"
            "    cell_indices: 1D int32 tensor of cell indices.\n\n"
            "Returns:\n"
            "    An (N, 8, 3) float32 tensor of corner coordinates.")
        .def(
            "filter_cell_indices",
            [](SparseGrid &self, SparseGridCellIndices cell_indices,
               SparseGridData values, float level = 0.f) {
                NDArray<uint> cell_indices_ =
                    nb_to_ours(cell_indices).cast<uint>();
                NDArray<float> values_ = nb_to_ours(values);
                NDArray<int> filtered_cell_indices =
                    self.filter_cell_indices(cell_indices_, values_, level)
                        .cast<int>();
                return ours_to_nb(filtered_cell_indices);
            },
            "cell_indices"_a, "values"_a, "level"_a = 0.f,
            "Filter cells to keep only those that cross the iso-surface.\n\n"
            "Args:\n"
            "    cell_indices: 1D int32 tensor of cell indices to filter.\n"
            "    values: (N, 8) float32 tensor of scalar values at cell corners.\n"
            "    level: The iso-value to check against. Default is 0.0.\n\n"
            "Returns:\n"
            "    A 1D int32 tensor of cell indices that cross the iso-surface.");

    nb::class_<Intersection>(m, "Intersection",
        "Stores edge-surface intersection points and normals for dual contouring.\n\n"
        "Created by get_intersection() and used as input to dual_contouring().\n"
        "You can modify the normals to control the surface reconstruction.")
        .def("get_points",
             [](Intersection &self) { return ours_to_nb(self.points); },
             "Return the intersection points as an (N, 3) float32 tensor.")
        .def("get_normals",
             [](Intersection &self) { return ours_to_nb(self.normals); },
             "Return the surface normals at intersection points as an (N, 3) float32 tensor.")
        .def(
            "set_normals",
            [](Intersection &self, Vector3 new_normals) {
                NDArray<float3> normals = nb_to_ours(new_normals);
                self.set_normals(normals);
            },
            "new_normals"_a,
            "Set custom normals for the intersection points.\n\n"
            "Args:\n"
            "    new_normals: (N, 3) float32 tensor of normal vectors.");

    m.def("get_intersection", &get_intersection, "grid"_a, "level"_a = 0.f,
        "Compute edge-surface intersections for dual contouring.\n\n"
        "Finds where grid edges cross the iso-surface and computes intersection\n"
        "points and normals using linear interpolation and central differences.\n\n"
        "Args:\n"
        "    grid: The input grid containing scalar values.\n"
        "    level: The iso-value. Default is 0.0.\n\n"
        "Returns:\n"
        "    An Intersection object containing points and normals.");

    m.def(
        "dual_contouring",
        [](Grid *grid, Intersection its, float level = 0.f, float reg = 1e-2f,
           float svd_tol = 1e-6f) {
            auto [v, f] = dual_contouring(grid, its, level, reg, svd_tol);
            return nb::make_tuple(ours_to_nb(v), ours_to_nb(f));
        },
        "grid"_a, "its"_a, "level"_a = 0.f, "reg"_a = 1e-2f,
        "svd_tol"_a = 1e-6f,
        "Extract an iso-surface using the Dual Contouring algorithm.\n\n"
        "Dual Contouring produces meshes with better-placed vertices than Marching Cubes,\n"
        "especially for sharp features. It solves a QEF (Quadric Error Function) per cell\n"
        "to find optimal vertex positions.\n\n"
        "Args:\n"
        "    grid: The input grid containing scalar values.\n"
        "    its: Intersection data from get_intersection().\n"
        "    level: The iso-value. Default is 0.0.\n"
        "    reg: Regularization weight for the QEF solver. Default is 0.01.\n"
        "    svd_tol: SVD tolerance for the QEF solver. Default is 1e-6.\n\n"
        "Returns:\n"
        "    A tuple (vertices, faces) where vertices is an (N, 3) float32 tensor\n"
        "    and faces is an (M, 4) uint32 tensor of quad indices.");

    m.doc() = "GPU-accelerated iso-surface extraction algorithms.\n\n"
              "This module provides implementations of Marching Cubes and Dual Contouring\n"
              "for extracting surfaces from scalar fields stored on uniform or sparse grids.\n"
              "All operations run on CUDA and use PyTorch tensors for data exchange.";
}
