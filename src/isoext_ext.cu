#include "mc.cuh"
#include "utils.cuh"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

// Function to create a nanobind capsule for host memory
nanobind::capsule
create_host_capsule(void *ptr) {
    return nanobind::capsule(ptr, [](void *p) noexcept { free(p); });
}

// Function to create a nanobind capsule for device memory
nanobind::capsule
create_device_capsule(void *ptr) {
    return nanobind::capsule(ptr, [](void *p) noexcept { cudaFree(p); });
}

NB_MODULE(isoext_ext, m) {

    m.def("marching_cubes", &MarchingCubes<nb::pytorch, nb::device::cuda>::run,
          "grid"_a, "aabb"_a, "level"_a = 0.f, "tight"_a = true,
          "method"_a = "lorensen", "Marching Cubes");
    m.def("marching_cubes", &MarchingCubes<nb::pytorch, nb::device::cpu>::run,
          "grid"_a, "aabb"_a, "level"_a = 0.f, "tight"_a = true,
          "method"_a = "lorensen", "Marching Cubes");
    m.def("marching_cubes", &MarchingCubes<nb::numpy, nb::device::cpu>::run,
          "grid"_a, "aabb"_a, "level"_a = 0.f, "tight"_a = true,
          "method"_a = "lorensen", "Marching Cubes");

    m.doc() = "A library for extracting iso-surfaces from level-set functions";
}
