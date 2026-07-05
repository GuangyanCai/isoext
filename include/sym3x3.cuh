#pragma once

#include "math.cuh"

// Closed-form solvers for symmetric 3x3 systems.
//
// Dual contouring solves one tiny QEF system per cell. Doing that in closed
// form inside the kernel is orders of magnitude faster than a batched
// cusolver SVD, which used to account for ~95% of the dual contouring
// runtime.

// Eigendecomposition A = V diag(eigenvalues) V^T of a symmetric 3x3 matrix
// by cyclic Jacobi rotations. A is destroyed in the process; the columns of
// V are the eigenvectors. A fixed number of sweeps is enough to reach float
// precision for any symmetric 3x3 matrix.
__host__ __device__ inline void
sym_eigen_3x3(float A[3][3], float eigenvalues[3], float V[3][3]) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            V[r][c] = (r == c) ? 1.0f : 0.0f;
        }
    }

    const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int sweep = 0; sweep < 8; sweep++) {
        for (int k = 0; k < 3; k++) {
            int p = pairs[k][0];
            int q = pairs[k][1];
            float apq = A[p][q];
            if (fabsf(apq) < 1e-20f) {
                continue;
            }

            // Rotation angle that zeroes A[p][q] (Golub & Van Loan).
            float theta = 0.5f * (A[q][q] - A[p][p]) / apq;
            float t = copysignf(1.0f, theta) /
                      (fabsf(theta) + sqrtf(theta * theta + 1.0f));
            float c = 1.0f / sqrtf(t * t + 1.0f);
            float s = t * c;

            // Apply the rotation to A; only rows/columns p, q and the
            // remaining index r change.
            int r = 3 - p - q;
            float arp = A[r][p];
            float arq = A[r][q];
            A[p][p] -= t * apq;
            A[q][q] += t * apq;
            A[p][q] = A[q][p] = 0.0f;
            A[r][p] = A[p][r] = c * arp - s * arq;
            A[r][q] = A[q][r] = s * arp + c * arq;

            // Accumulate the rotation into the eigenvectors.
            for (int i = 0; i < 3; i++) {
                float vip = V[i][p];
                float viq = V[i][q];
                V[i][p] = c * vip - s * viq;
                V[i][q] = s * vip + c * viq;
            }
        }
    }

    eigenvalues[0] = A[0][0];
    eigenvalues[1] = A[1][1];
    eigenvalues[2] = A[2][2];
}

// Solve A x = b for a symmetric positive semi-definite A via the truncated
// pseudo-inverse: eigenvalues below tol times the largest one are treated
// as zero, dropping the contribution of the corresponding directions.
__host__ __device__ inline float3
solve_sym_3x3(const float A[3][3], float3 b, float tol) {
    float M[3][3];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            M[r][c] = A[r][c];
        }
    }
    float eigenvalues[3];
    float V[3][3];
    sym_eigen_3x3(M, eigenvalues, V);

    float threshold = tol * fmaxf(eigenvalues[0],
                                  fmaxf(eigenvalues[1], eigenvalues[2]));

    float3 x = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 3; i++) {
        if (eigenvalues[i] > threshold) {
            float3 v = make_float3(V[0][i], V[1][i], V[2][i]);
            x = x + (dot(v, b) / eigenvalues[i]) * v;
        }
    }
    return x;
}
