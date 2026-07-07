# isoext

[![PyPI version](https://badge.fury.io/py/isoext.svg?)](https://badge.fury.io/py/isoext)
[![GitHub](https://img.shields.io/badge/GitHub-isoext-blue?logo=github)](https://github.com/GuangyanCai/isoext)
[![License](https://img.shields.io/github/license/GuangyanCai/isoext)](https://github.com/GuangyanCai/isoext/blob/master/LICENSE)

**GPU-accelerated iso-surface extraction for PyTorch**

```{raw} html
<div class="hero-viewer">
  <iframe src="_static/viser/index.html?playbackPath=../scenes/hero.viser&darkMode&initialCameraPosition=1.5,1.5,1.1&initialCameraLookAt=0,0,0&initialCameraUp=0,0,1"
          frameborder="0"></iframe>
  <p class="hero-caption">
    A power-8 Mandelbulb, extracted with <code>isoext.marching_cubes</code>:
    around 220k vertices in about 2&nbsp;ms. Drag to rotate.
  </p>
</div>
```

```{tip}
Every 3D view in these docs is interactive: drag to orbit, right-drag
to pan, scroll to zoom. The {doc}`viewer page <viewer>` covers the
controls and how to open one from your own code.
```

An iso-surface is the set of points where a 3D scalar field equals a
chosen value: the shape described by a signed distance function, or the
boundary of a density volume. `isoext` is a growing collection of
iso-surface extraction methods that turn such fields into triangle
meshes on the GPU. The field values come in as a PyTorch tensor and the
mesh comes back as tensors, so it fits directly into training loops and
other GPU pipelines. {doc}`how_it_works` explains and compares the
methods.

## Features

- **Extraction methods** — sharing one grid interface, with more on the way
  - {doc}`Marching Cubes <marching_cubes>` — supports the same
    topology-correct MC33 as scikit-image (`lewiner`) and defaults to an
    improved variant with the corrected interior test (`vega`); see
    {doc}`mc_variants`
  - {doc}`Marching Tetrahedra <marching_tetrahedra>` — ambiguity-free
    extraction by splitting cells into tetrahedra
  - {doc}`Dual Contouring <dual_contouring>` — sharp feature preservation
    from surface normals
  - {doc}`Surface Nets <surface_nets>` — smooth dual meshes without
    needing normals
- {doc}`Grids <grids>`
  - Dense uniform grids for full volumes
  - Sparse grids that only store cells near the surface, so memory scales
    with area instead of volume
- {doc}`Interactive viewer <viewer>` — meshes and grid overlays in the
  browser, built on [viser](https://viser.studio); scenes can be embedded
  in static web pages
- **SDF toolbox** — primitives from spheres to a Mandelbulb, CSG
  operations, and gradient and smoothing utilities ({doc}`sdf_guide`)

## Quick Example

```python
import isoext
from isoext import viewer

grid = isoext.UniformGrid([256, 256, 256])
grid.set_values(grid.get_points().norm(dim=-1) - 0.8)  # Sphere

vertices, faces = isoext.marching_cubes(grid)

server = viewer.show(vertices, faces)  # opens the mesh in the browser
isoext.write_obj("sphere.obj", vertices, faces)
```

## Performance

Median extraction times for a sphere SDF on an RTX 5090:

| Algorithm           | uniform 512³ | sparse 512³ |
|---------------------|--------------|-------------|
| marching_cubes      | 5.4 ms       | 1.9 ms      |
| marching_tetrahedra | 6.7 ms       | 3.2 ms      |
| dual_contouring     | 7.0 ms       | 2.3 ms      |
| surface_nets        | 6.9 ms       | 2.2 ms      |

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
marching_tetrahedra
dual_contouring
surface_nets
```

```{toctree}
:maxdepth: 2
:caption: In Depth
:hidden:

how_it_works
mc_variants
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
:caption: Visualization
:hidden:

viewer
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api
performance
references
```

```{toctree}
:maxdepth: 2
:caption: Development
:hidden:

development
```
