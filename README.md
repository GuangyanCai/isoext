
# isoext: Isosurface Extraction on GPU
[![PyPI version](https://badge.fury.io/py/isoext.svg?kill_cache=1)](https://badge.fury.io/py/isoext)
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
sdf_v = sdf(grid) # Only accept a gpu tensor from pytorch for now

isolevel = 0

v, f = isoext.marching_cubes(sdf_v, aabb, isolevel)

isoext.write_obj('test.obj', v, f)
```

## Task List
- [x] Fix docstring.
- [ ] Implement MC33.
- [x] Add `numpy` support.

## License

`isoext` is released under the [MIT License](LICENSE). Feel free to use it in your projects.

## Acknowledgments
* We use [Thrust](https://developer.nvidia.com/thrust) for GPU computing and [nanobind](https://github.com/wjakob/nanobind) for Python binding. 
* The LUTs for `lorensen` are borrowed from Paul Bourke (https://paulbourke.net/geometry/polygonise/).
