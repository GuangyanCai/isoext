#include <nanobind/nanobind.h>
#include <test.hpp>

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(diff_voxel_ext, m) {
    m.def("add", add);
}
