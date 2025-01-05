
# isoext: Isosurface Extraction on GPU
[![Documentation](https://img.shields.io/badge/docs-blue.svg)](https://github.com/GuangyanCai/isoext/wiki)
[![PyPI version](https://badge.fury.io/py/isoext.svg?)](https://badge.fury.io/py/isoext)

## Overview

`isoext` is a high-performance Python library for GPU-accelerated isosurface extraction.

### ✨ Key Features

🔷 **Different Isosurface Extraction Methods**
- Marching Cubes
  - `lorensen`: the original marching cubes algorithm from the paper [Marching cubes: A high resolution 3D surface construction algorithm](https://dl.acm.org/doi/10.1145/37402.37422).
  - `nagae`: the marching cubes algorithm from the paper [Surface construction and contour generation from volume data](https://doi.org/10.1117/12.154567). It uses only rotation to transform the Marching Cubes cases, unlike `lorensen` which uses rotation and reflection. `lorensen` contains ambiguities which results in holes and cracks. This modification removes the ambiguities and produces a closed surface.
- Dual Contouring
  - `dual_contouring`: the dual contouring algorithm from the paper [Dual Contouring of Hermite Data](https://dl.acm.org/doi/10.1145/566654.566586).
- More methods will be added in the future.

🔷 **Flexible Grid Support** 
- Uniform grid for regular sampling
- Sparse grid for memory efficiency
- Octree grid is coming soon

🔷 **Developer Tools**
- Built-in Marching Cubes table generator
  - All lookup tables used in the library are generated using this tool
- Rich set of SDF primitives and operators
  - Create custom SDFs by combining primitives and operators.

## Installation

`isoext` currently requires PyTorch with CUDA support for GPU acceleration.

### Prerequisites
- [PyTorch](https://pytorch.org/) with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) matching your PyTorch version
- A compatible C++ compiler (e.g., Visual Studio on Windows, GCC on Linux)

### Install from PyPI
The simplest way to install `isoext` is via pip:
```bash
pip install isoext
```
### Install from Source

```bash
git clone https://github.com/GuangyanCai/isoext
cd isoext
pip install .
```

## Quick Start

Here's a simple example to get you started:

```python
import isoext
from isoext.sdf import *

# Create grid
grid = isoext.UniformGrid([256, 256, 256])

# Create composite SDF shape - a sphere with three orthogonal toroidal holes
torus_a = TorusSDF(R=0.75, r=0.15)  # Base torus in xy plane
torus_b = RotationOp(sdf=torus_a, axis=[1, 0, 0], angle=90)  # Rotated to xz plane
torus_c = RotationOp(sdf=torus_a, axis=[0, 1, 0], angle=90)  # Rotated to yz plane
sphere_a = SphereSDF(radius=0.75)
sdf = IntersectionOp([sphere_a, NegationOp(UnionOp([torus_a, torus_b, torus_c]))])

# Evaluate SDF and extract isosurface
sdf_v = sdf(grid.get_points())
grid.set_values(sdf_v)

# Run marching cubes
print("Running marching cubes")
v, f = isoext.marching_cubes(grid)
print("Writing obj")
isoext.write_obj("mc.obj", v, f)
print("Done")

# Run dual contouring
print("Running dual contouring")
its = isoext.get_intersection(grid)
points = its.get_points()
normals = get_sdf_normal(sdf, points)
its.set_normals(normals)
v, f = isoext.dual_contouring(grid, its)
print("Writing obj")
isoext.write_obj("dc.obj", v, f)
print("Done")
```

## Documentation

The documentation is available on the [wiki](https://github.com/GuangyanCai/isoext/wiki).

## Future Plans
- [ ] Add Dual Marching Cubes.
- [ ] Other more recent isosurface extraction methods.
- [ ] Support more libraries such as `numpy` and `jax`.

## License

`isoext` is released under the [MIT License](LICENSE). Feel free to use it in your projects.

## Acknowledgments
We use the following libraries:
* [Thrust](https://developer.nvidia.com/thrust) for GPU computing.
* [cuBLAS](https://developer.nvidia.com/cublas) and [cuSOLVER](https://developer.nvidia.com/cusolver) for QEF solving.
* [nanobind](https://github.com/wjakob/nanobind) for Python binding.