#include "utils.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

template<typename T>
T* device_to_host(const T* d_ptr, size_t size) {
    T* h_ptr = (T*) malloc(size * sizeof(T));
    cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    return h_ptr;
}

template<typename T>
T* host_to_device(const T* h_ptr, size_t size){
    T* d_ptr = (T*) malloc(size * sizeof(T));   
    cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    return d_ptr;
}

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