
# isoext: Isosurface Extraction on GPU
[![PyPI version](https://badge.fury.io/py/isoext.svg?)](https://badge.fury.io/py/isoext)
## Overview

Welcome to `isoext` — a Python library designed for efficient isosurface extraction, leveraging the power of GPU computing and comes with `pytorch` support. Our library attempts to implement a collection of classic isosurface extraction algorithms. Currently, only the following algorithms are supported, but more will come in the future:
* Marching cubes
  * `lorensen`: the original marching cubes algorithm from the paper [Marching cubes: A high resolution 3D surface construction algorithm](https://dl.acm.org/doi/10.1145/37402.37422).
  * `nagae`: the marching cubes algorithm from the paper [Surface construction and contour generation from volume data](https://doi.org/10.1117/12.154567). It uses only rotation to transform the Marching Cubes cases, unlike `lorensen` which uses rotation and reflection. `lorensen` contains ambiguities which results in holes and cracks. This modification removes the ambiguities and produces a closed surface.

`isoext` also comes with a [Marching Cubes table generator](#marching-cubes-table-generator), which you can use to integrate Marching Cubes into your own project or extend `isoext` with more algorithms. All the lookup tables we used are generated by the generator.

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

v, f = isoext.marching_cubes(sdf_v, aabb=aabb, level=isolevel, method="nagae")

isoext.write_obj('test.obj', v, f)
```

## Marching Cubes

### Arguments

`isoext.marching_cubes` accepts the following arguments:

- `grid`: A CUDA PyTorch tensor representing the scalar field. It should be a 3D tensor. If `cells` is provided, `grid` must be of shape (2N, 2, 2), where N is the number of cells.
- `aabb`: (Optional) A list or tuple of 6 floats representing the axis-aligned bounding box [xmin, ymin, zmin, xmax, ymax, zmax]. If provided, `cells` must not be given.
- `cells`: (Optional) A CUDA PyTorch tensor of shape (2N, 2, 2, 3) representing the cell positions. If provided, `aabb` must not be given.
- `level`: (Optional) The isovalue at which to extract the isosurface. Default is 0.0.
- `method`: (Optional) The marching cubes algorithm to use. Currently, `lorensen` and `nagae` are supported. Default is `nagae`.

### Return Value

The function returns a tuple `(vertices, faces)`:

- `vertices`: A CUDA PyTorch tensor of shape (V, 3) representing the vertex positions. 
- `faces`: A CUDA PyTorch tensor of shape (F, 3) representing the triangular faces.
- If no faces are found, both `vertices` and `faces` will be `None`.


## Marching Cubes Table Generator
You can find the generator in `luts/gen_mc_lut.py`. Available methods are in `luts/mc_methods`. For example, to generate the table for `lorensen`, run:

```bash
cd luts
python gen_mc_lut.py mc_methods/lorensen.json output
```
This will generate the luts and as well as the meshes of all the cases inside the `output/lorensen` folder.

For the cube annotation, refer to the script documentation in `luts/gen_mc_lut.py`. The cases are indexed in binary format, where the ith bit indicates the ith vertex is below the isosurface if it is 1 and vice versa.

## Task List
- [x] Fix docstring.
- [ ] Add more Marching Cubes variants.
- [ ] Add Dual Contouring.
- [ ] Add Dual Marching Cubes.

## License

`isoext` is released under the [MIT License](LICENSE). Feel free to use it in your projects.

## Acknowledgments
* We use [Thrust](https://developer.nvidia.com/thrust) for GPU computing and [nanobind](https://github.com/wjakob/nanobind) for Python binding. 