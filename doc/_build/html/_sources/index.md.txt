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
    <code>isoext.marching_cubes</code>: 225k vertices in under 10&nbsp;ms.
    Drag to orbit.
  </p>
</div>
```

`isoext` turns scalar fields into meshes without leaving the GPU: feed it a
PyTorch tensor of field values — from an analytic SDF, a neural network, or
an occupancy volume — and get vertices and faces back as tensors.

```{raw} html
<div class="feature-grid">
  <div class="feature-card"><strong>Marching Cubes</strong>
    Fast triangular mesh extraction from dense or sparse grids.</div>
  <div class="feature-card"><strong>Dual Contouring</strong>
    QEF-based vertex placement that preserves sharp features.</div>
  <div class="feature-card"><strong>Sparse Grids</strong>
    Track only surface-crossing cells; cost scales with area, not volume.</div>
  <div class="feature-card"><strong>Interactive Viewer</strong>
    One-line mesh inspection in the browser, built on viser.</div>
</div>
```

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

## Sharp features, side by side

The same 81³ Menger sponge field, extracted by both algorithms. Marching
cubes chamfers the edges; dual contouring keeps them crisp.

```{raw} html
<div class="compare-grid">
  <figure>
    <iframe src="_static/viser/index.html?playbackPath=../scenes/menger_mc.viser&darkMode&initialCameraPosition=1.9,1.4,1.2&initialCameraLookAt=0,0,0&initialCameraUp=0,0,1"
            frameborder="0"></iframe>
    <figcaption>marching_cubes</figcaption>
  </figure>
  <figure>
    <iframe src="_static/viser/index.html?playbackPath=../scenes/menger_dc.viser&darkMode&initialCameraPosition=1.9,1.4,1.2&initialCameraLookAt=0,0,0&initialCameraUp=0,0,1"
            frameborder="0"></iframe>
    <figcaption>dual_contouring</figcaption>
  </figure>
</div>
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
:caption: User Guide
:hidden:

grids
marching_cubes
dual_contouring
sdf_guide
```

```{toctree}
:maxdepth: 2
:caption: Extras
:hidden:

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
