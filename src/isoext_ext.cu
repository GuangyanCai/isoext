#include "mc.cuh"
#include "utils.cuh"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include <thrust/device_malloc.h>

#include <cstdint>
#include <algorithm>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(isoext_ext, m) {

    m.def("marching_cubes", &MarchingCubes<nb::pytorch, nb::device::cuda>::run, "grid"_a, "aabb"_a, "level"_a = 0.f, "method"_a = "lorensen", "Marching Cubes");
    m.def("marching_cubes", &MarchingCubes<nb::pytorch, nb::device::cpu>::run, "grid"_a, "aabb"_a, "level"_a = 0.f, "method"_a = "lorensen", "Marching Cubes");
    m.def("marching_cubes", &MarchingCubes<nb::numpy, nb::device::cpu>::run, "grid"_a, "aabb"_a, "level"_a = 0.f, "method"_a = "lorensen", "Marching Cubes");

    m.doc() = "A library for extracting iso-surfaces from level-set functions";
    
}
