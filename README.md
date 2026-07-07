# isoext

[![PyPI version](https://badge.fury.io/py/isoext.svg?)](https://badge.fury.io/py/isoext)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://guangyancai.github.io/isoext/)
[![License](https://img.shields.io/github/license/GuangyanCai/isoext)](LICENSE)

**GPU-accelerated iso-surface extraction for PyTorch**

An iso-surface is the set of points where a 3D scalar field equals a chosen value: the shape described by a signed distance function, or the boundary of a density volume. `isoext` is a growing collection of iso-surface extraction methods that turn such fields into triangle meshes on the GPU, taking the field in as a PyTorch tensor and returning the mesh as tensors.

## Features

- **Extraction methods** — sharing one grid interface, with more on the way
  - **Marching Cubes** — supports the same topology-correct MC33 as scikit-image (`lewiner`) and defaults to an improved variant with the corrected interior test (`vega`); the classic `nagae` and `lorensen` tables are included too
  - **Marching Tetrahedra** — ambiguity-free extraction by splitting cells into tetrahedra
  - **Dual Contouring** — sharp feature preservation from surface normals
  - **Surface Nets** — smooth dual meshes without needing normals
  - **Dual Marching Cubes** — sharp features with one vertex per surface sheet, so crossing sheets stay separate
- **Grids**
  - Dense uniform grids for full volumes
  - Sparse grids that only store cells near the surface, so memory scales with area instead of volume
- **Interactive viewer** — meshes and grid overlays in the browser, built on [viser](https://viser.studio); scenes can be embedded in static web pages
- **SDF toolbox** — primitives from spheres to a Mandelbulb, CSG operations, and gradient and smoothing utilities

## Installation

Requires PyTorch with CUDA support; building from source needs CUDA 12.4+.

```bash
pip install isoext
```

## Quick Start

```python
import isoext
from isoext import viewer

grid = isoext.UniformGrid([256, 256, 256])
grid.set_values(grid.get_points().norm(dim=-1) - 0.8)  # Sphere

vertices, faces = isoext.marching_cubes(grid)

server = viewer.show(vertices, faces)  # opens the mesh in the browser
isoext.write_obj("sphere.obj", vertices, faces)
```

## Documentation

See the [full documentation](https://guangyancai.github.io/isoext/) for guides on grids, extraction methods, and the API reference.

## License

MIT License. See [LICENSE](LICENSE) for details.
