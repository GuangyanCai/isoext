# isoext

[![PyPI version](https://badge.fury.io/py/isoext.svg?)](https://badge.fury.io/py/isoext)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://guangyancai.github.io/isoext/)

**GPU-accelerated iso-surface extraction for PyTorch**

`isoext` is a high-performance library for extracting surfaces from scalar fields using CUDA.

## Features

- **Marching Cubes** — Fast triangular mesh extraction (~5 ms for a 512³ grid on an RTX 5090)
- **Dual Contouring** — QEF-based vertex placement that preserves sharp features
- **Flexible Grids** — Dense uniform grids and sparse grids whose cost scales with surface area, not volume
- **Interactive Viewer** — One-line mesh inspection in the browser, built on [viser](https://viser.studio)
- **SDF Utilities** — Primitives and CSG operations for building test fields

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

server = viewer.show(vertices, faces)  # interactive viewer in the browser
isoext.write_obj("sphere.obj", vertices, faces)
```

## Documentation

See the [full documentation](https://guangyancai.github.io/isoext/) for guides on grids, extraction methods, and the API reference.

## License

MIT License. See [LICENSE](LICENSE) for details.
