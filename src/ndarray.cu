#include "ndarray.cuh"

#include <iostream>
#include <string>

void
print_batched_matrix(const NDArray<float> &arr, const std::string &name) {
    if (arr.ndim() != 3) {
        throw std::runtime_error("Input matrix must be 3D");
    }
    const int batch = arr.shape[0];
    const int m = arr.shape[1];
    const int n = arr.shape[2];

    // Copy data to host
    std::vector<float> host_data(arr.size());
    thrust::copy(arr.data_ptr, arr.data_ptr + arr.size(), host_data.begin());
    // Print name
    std::cout << name << ":\n";
    // Print each batch
    for (int b = 0; b < batch; b++) {
        std::cout << "Batch " << b << ":\n";
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << host_data[b * m * n + i * n + j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void
print_batched_vector(const NDArray<float> &arr, const std::string &name) {
    if (arr.ndim() != 2) {
        throw std::runtime_error("Input vector must be 2D");
    }
    const int batch = arr.shape[0];
    const int n = arr.shape[1];

    // Copy data to host
    std::vector<float> host_data(arr.size());
    thrust::copy(arr.data_ptr, arr.data_ptr + arr.size(), host_data.begin());

    // Print name
    std::cout << name << ":\n";
    // Print each batch
    for (int b = 0; b < batch; b++) {
        std::cout << "Batch " << b << ": ";
        for (int i = 0; i < n; i++) {
            std::cout << host_data[b * n + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}