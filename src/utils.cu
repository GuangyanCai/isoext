#include "utils.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

nanobind::capsule create_host_capsule(void* ptr) {
    return nanobind::capsule(ptr, [](void* p) noexcept {
        free(p);
    });
}

nanobind::capsule create_device_capsule(void* ptr) {
    return nanobind::capsule(ptr, [](void* p) noexcept {
        cudaFree(p);
    });
}