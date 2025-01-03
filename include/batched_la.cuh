#pragma once

#include "ndarray.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <tuple>

class BatchedLASolver {
  private:
    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;

  public:
    BatchedLASolver();
    ~BatchedLASolver();

    std::tuple<NDArray<float>, NDArray<float>, NDArray<float>, NDArray<int>>
    svd(const NDArray<float> &A);

    NDArray<float> gemm(const NDArray<float> &A, const NDArray<float> &B,
                        bool transa = false, bool transb = false);

    NDArray<float> gemv(const NDArray<float> &A, const NDArray<float> &x,
                        bool transa = false);

    std::tuple<NDArray<float>, NDArray<int>> lsq_svd(const NDArray<float> &A,
                                                     const NDArray<float> &b,
                                                     float threshold = 1e-7);
};