#include "batched_la.cuh"
#include "ndarray.cuh"

#include <iostream>
#include <string>

BatchedLASolver::BatchedLASolver() {
    cusolverDnCreate(&cusolver_handle);
    cublasCreate(&cublas_handle);
}

BatchedLASolver::~BatchedLASolver() {
    cusolverDnDestroy(cusolver_handle);
    cublasDestroy(cublas_handle);
}

std::tuple<NDArray<float>, NDArray<float>, NDArray<float>, NDArray<int>>
BatchedLASolver::svd(const NDArray<float> &A) {
    assert(A.ndim() == 3);
    const int batch = A.shape[0];
    const int m = A.shape[1];
    const int n = A.shape[2];
    const int lda = m;   // leading dimension of A
    const int ldu = m;   // leading dimension of U
    const int ldv = n;   // leading dimension of V
    NDArray<float> U = NDArray<float>::zeros(batch, m, m);
    NDArray<float> VT = NDArray<float>::zeros(batch, n, n);
    NDArray<float> S = NDArray<float>::zeros(batch, n);

    // Configure gesvdj parameters
    gesvdjInfo_t gesvdj_params;
    cusolverDnCreateGesvdjInfo(&gesvdj_params);

    // Get workspace size needed
    int work_size = 0;
    cusolverDnSgesvdj_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR,
                                 1,   // compute singular vectors
                                 m, n, A.data(), lda, S.data(), U.data(), lda,
                                 VT.data(), n, &work_size, gesvdj_params);

    NDArray<float> work = NDArray<float>::zeros(work_size);

    NDArray<int> info = NDArray<int>::zeros(batch);

    cusolverDnSgesvdjBatched(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, m, n,
                             A.data(), lda, S.data(), U.data(), ldu, VT.data(),
                             ldv, work.data(), work_size, info.data(),
                             gesvdj_params, batch);

    cusolverDnDestroyGesvdjInfo(gesvdj_params);

    return {U, VT, S, info};
}

NDArray<float>
BatchedLASolver::gemm(const NDArray<float> &A, const NDArray<float> &B,
                      bool transa, bool transb) {
    assert(A.ndim() == 3 && B.ndim() == 3);
    assert(A.shape[0] == B.shape[0]);   // same batch size
    const int batch = A.shape[0];

    // Get matrix dimensions based on transpose flags
    const int m = transa ? A.shape[2] : A.shape[1];
    const int k = transa ? A.shape[1] : A.shape[2];
    const int n = transb ? B.shape[1] : B.shape[2];

    // Verify matrix dimensions are compatible
    assert(k == (transb ? B.shape[2] : B.shape[1]));

    // Leading dimensions
    const int lda = A.shape[1];   // Leading dimension of A
    const int ldb = B.shape[1];   // Leading dimension of B
    const int ldc = m;            // Leading dimension of output

    // Create output matrix
    NDArray<float> C = NDArray<float>::zeros(batch, m, n);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Set transpose operations
    cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Perform batched matrix multiplication
    cublasSgemmStridedBatched(
        cublas_handle, opA, opB, m, n, k, &alpha, A.data(), lda,
        lda * k,                         // A, leading dim, stride
        B.data(), ldb, ldb * n,          // B, leading dim, stride
        &beta, C.data(), ldc, ldc * n,   // C, leading dim, stride
        batch);

    return C;
}

NDArray<float>
BatchedLASolver::gemv(const NDArray<float> &A, const NDArray<float> &x,
                      bool transa) {
    assert(A.ndim() == 3 && x.ndim() == 2);
    assert(A.shape[0] == x.shape[0]);   // same batch size
    const int batch = A.shape[0];

    // Get matrix dimensions based on transpose flag
    const int m = transa ? A.shape[2] : A.shape[1];
    const int n = transa ? A.shape[1] : A.shape[2];

    // Verify vector dimension is compatible
    assert(x.shape[1] == n);

    // Leading dimensions
    const int lda = A.shape[1];   // Leading dimension of A
    const int incx = 1;           // Increment for x
    const int incy = 1;           // Increment for y

    // Create output vector
    NDArray<float> y = NDArray<float>::zeros(batch, m);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Set transpose operation
    cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Perform batched matrix-vector multiplication
    cublasSgemvStridedBatched(
        cublas_handle, opA, A.shape[1],
        A.shape[2],                                // m, n (original dimensions)
        &alpha, A.data(), lda, lda * A.shape[2],   // A, leading dim, stride
        x.data(), incx, n,                         // x, increment, stride
        &beta, y.data(), incy, m,                  // y, increment, stride
        batch);

    return y;
}

std::tuple<NDArray<float>, NDArray<int>>
BatchedLASolver::lsq_svd(const NDArray<float> &A, const NDArray<float> &b,
                         float tol) {
    // Compute SVD
    auto [U, VT, S, info] = svd(A);
    // Compute U^T * b
    auto UTb = gemv(U, b, true);

    // Inverse S with threshold for numerical stability
    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(S.shape[0]),
                     [tol, S = S.data(), len = S.shape[1]] __device__(int i) {
                         int offset = i * len;
                         float threshold = tol * S[offset];
                         for (int j = 0; j < len; ++j) {
                             S[offset + j] = S[offset + j] > threshold
                                                 ? 1.0f / S[offset + j]
                                                 : 0.0f;
                         }
                     });
    // Multiply by S^-1
    NDArray<float> S_inv_UTb({S.shape.begin(), S.shape.end()});
    thrust::transform(S.data_ptr, S.data_ptr + S.size(), UTb.data_ptr,
                      S_inv_UTb.data_ptr,
                      [] __device__(float s, float utb) { return s * utb; });
    // Multiply by V^T
    NDArray<float> x = gemv(VT, S_inv_UTb);
    return {x, info};
}