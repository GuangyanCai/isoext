# isoext

[![PyPI version](https://badge.fury.io/py/isoext.svg?)](https://badge.fury.io/py/isoext)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://guangyancai.github.io/isoext/)

**GPU-accelerated iso-surface extraction for PyTorch**

`isoext` is a high-performance library for extracting surfaces from scalar fields using CUDA.

## Features

- **Marching Cubes** — Fast triangular mesh extraction
- **Dual Contouring** — Triangle meshes with sharp feature preservation
- **Flexible Grids** — Dense uniform grids and memory-efficient sparse grids
- **SDF Utilities** — Optional primitives and CSG operations

## Installation

Requires PyTorch with CUDA support, as well as the matching CUDA compiler.

```bash
pip install isoext
```

## Quick Start

```python
import isoext

grid = isoext.UniformGrid([256, 256, 256])
grid.set_values(grid.get_points().norm(dim=-1) - 0.8)  # Sphere

vertices, faces = isoext.marching_cubes(grid)
isoext.write_obj("sphere.obj", vertices, faces)
```

## Documentation

See the [full documentation](https://guangyancai.github.io/isoext/) for guides on grids, extraction methods, and the API reference.

## License

MIT License. See [LICENSE](LICENSE) for details.
