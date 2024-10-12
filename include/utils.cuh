#pragma once

#include <nanobind/nanobind.h>

template<typename T>
T* device_to_host(const T* d_ptr, size_t size);

template<typename T>
T* host_to_device(const T* h_ptr, size_t size);

nanobind::capsule create_host_capsule(void* ptr);

nanobind::capsule create_device_capsule(void* ptr);