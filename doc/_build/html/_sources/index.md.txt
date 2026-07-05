# isoext

[![PyPI version](https://badge.fury.io/py/isoext.svg?)](https://badge.fury.io/py/isoext)
[![GitHub](https://img.shields.io/badge/GitHub-isoext-blue?logo=github)](https://github.com/GuangyanCai/isoext)

**GPU-accelerated iso-surface extraction for PyTorch**

```{raw} html
<div class="hero-viewer">
  <iframe src="_static/viser/index.html?playbackPath=../scenes/hero.viser&darkMode&initialCameraPosition=1.5,1.5,1.1&initialCameraLookAt=0,0,0&initialCameraUp=0,0,1"
          frameborder="0"></iframe>
  <p class="hero-caption">
    Power-8 Mandelbulb &mdash; a raw PyTorch tensor field meshed by
    <code>isoext.marching_cubes</code>: 219k vertices in ~4&nbsp;ms.
    Drag to orbit.
  </p>
</div>
```

`isoext` turns scalar fields into meshes without leaving the GPU: feed it a
PyTorch tensor of field values — from an analytic SDF, a neural network, or
an occupancy volume — and get vertices and faces back as tensors. Extraction
methods include marching cubes and dual contouring, on both dense and
sparse grids, with an interactive browser viewer built in.

## Quick Example

```python
import isoext
from isoext import viewer

grid = isoext.UniformGrid([256, 256, 256])
grid.set_values(grid.get_points().norm(dim=-1) - 0.8)  # Sphere

vertices, faces = isoext.marching_cubes(grid)

server = viewer.show(vertices, faces)  # interactive viewer in the browser
isoext.write_obj("sphere.obj", vertices, faces)
```

## Performance

Median extraction times for a sphere SDF on an RTX 5090:

| Algorithm       | uniform 512³ | sparse 512³ |
|-----------------|--------------|-------------|
| marching_cubes  | 4.8 ms       | 1.4 ms      |
| dual_contouring | 7.0 ms       | 2.3 ms      |

See {doc}`performance` for the full table and how to reproduce it.

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Extraction Methods
:hidden:

marching_cubes
dual_contouring
```

```{toctree}
:maxdepth: 2
:caption: Working with Fields
:hidden:

grids
sdf_guide
occupancy_grids
more_sdf
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api
performance
```
