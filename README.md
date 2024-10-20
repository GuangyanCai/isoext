
# isoext: Isosurface Extraction on GPU
[![PyPI version](https://badge.fury.io/py/isoext.svg?)](https://badge.fury.io/py/isoext)
## Overview

Welcome to `isoext` — a Python library designed for efficient isosurface extraction, leveraging the power of GPU computing and comes with `pytorch` support. Our library attempts to implement a collection of classic isosurface extraction algorithms. Currently, only the following algorithms are supported, but more will come in the future:
* Marching cubes
  * `lorensen`: the original marching cubes algorithm from the paper [Marching cubes: A high resolution 3D surface construction algorithm](https://dl.acm.org/doi/10.1145/37402.37422).

## Installation

To install `isoext`, make sure [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is installed and run:

```bash
pip install isoext
```

## Quick Start

Here's a simple example to get you started:

```python
import isoext
from isoext.sdf import *

aabb = [-1, -1, -1, 1, 1, 1]
res = 128
grid = isoext.make_grid(aabb, res)

torus_a = TorusSDF(R=0.75, r=0.15)
torus_b = RotationOp(sdf=torus_a, axis=[1, 0, 0], angle=90)
torus_c = RotationOp(sdf=torus_a, axis=[0, 1, 0], angle=90)

sphere_a = SphereSDF(radius=0.75)

sdf = IntersectionOp([
    sphere_a, 
    NegationOp(UnionOp([
        torus_a, torus_b, torus_c
    ]))
])
sdf_v = sdf(grid) # must be a cuda pytorchtensor

isolevel = 0

v, f = isoext.marching_cubes(sdf_v, aabb=aabb, level=isolevel, method="lorensen")

isoext.write_obj('test.obj', v, f)
```

## Marching Cubes

### Arguments

`isoext.marching_cubes` accepts the following arguments:

- `grid`: A CUDA PyTorch tensor representing the scalar field. It should be a 3D tensor. If `cells` is provided, `grid` must be of shape (2N, 2, 2), where N is the number of cells.
- `aabb`: (Optional) A list or tuple of 6 floats representing the axis-aligned bounding box [xmin, ymin, zmin, xmax, ymax, zmax]. If provided, `cells` must not be given.
- `cells`: (Optional) A CUDA PyTorch tensor of shape (2N, 2, 2, 3) representing the cell positions. If provided, `aabb` must not be given.
- `level`: The isovalue at which to extract the isosurface. Default is 0.0.
- `method`: The marching cubes algorithm to use. Currently, only "lorensen" is supported.

### Return Value

The function returns a tuple `(vertices, faces)`:

- `vertices`: A CUDA PyTorch tensor of shape (V, 3) representing the vertex positions. 
- `faces`: A CUDA PyTorch tensor of shape (F, 3) representing the triangular faces.
- If no faces are found, both `vertices` and `faces` will be `None`.

## Task List
- [x] Fix docstring.
- [ ] Add more Marching Cubes variants.
- [ ] Add Dual Contouring.
- [ ] Add Dual Marching Cubes.

## License

`isoext` is released under the [MIT License](LICENSE). Feel free to use it in your projects.

## Acknowledgments
* We use [Thrust](https://developer.nvidia.com/thrust) for GPU computing and [nanobind](https://github.com/wjakob/nanobind) for Python binding. 
* The LUTs for `lorensen` are borrowed from Paul Bourke (https://paulbourke.net/geometry/polygonise/).
