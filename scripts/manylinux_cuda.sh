#!/bin/bash
# Install a CUDA toolkit inside the manylinux_2_28 (AlmaLinux 8) build
# container, for cibuildwheel's before-all step. Usage: manylinux_cuda.sh 12.6
set -euo pipefail
version="$1"
pkg="${version/./-}"   # 12.6 -> 12-6

curl -fsSL -o /etc/yum.repos.d/cuda.repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
# The compiler, runtime headers and Thrust/CUB are all the extension needs.
dnf -y install "cuda-compiler-${pkg}" "cuda-cudart-devel-${pkg}" "cuda-cccl-${pkg}" \
    "cuda-driver-devel-${pkg}" gcc-toolset-13
ln -sfn "/usr/local/cuda-${version}" /usr/local/cuda
/usr/local/cuda/bin/nvcc --version | tail -1
