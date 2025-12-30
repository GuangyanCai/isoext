# isoext

**GPU-accelerated iso-surface extraction for PyTorch**

`isoext` is a high-performance library for extracting surfaces from scalar fields using CUDA. It provides implementations of Marching Cubes and Dual Contouring, optimized for GPU execution.

## Features

- **Marching Cubes** — Fast triangular mesh extraction
- **Dual Contouring** — Triangle mesh extraction with sharp feature preservation
- **Flexible Grids** — Dense uniform grids and memory-efficient sparse grids
- **SDF Utilities** — Optional primitives and CSG operations

## Quick Example

```python
import isoext

grid = isoext.UniformGrid([256, 256, 256])
grid.set_values(grid.get_points().norm(dim=-1) - 0.8)  # Sphere

vertices, faces = isoext.marching_cubes(grid)
```

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
:caption: Reference
:hidden:

api
```
