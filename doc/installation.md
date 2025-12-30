# Installation

## Prerequisites

`isoext` requires:

- **Python** 3.8 – 3.12
- **PyTorch** with CUDA support
- **CUDA Toolkit** matching your PyTorch version
- A C++ compiler (GCC on Linux, Visual Studio on Windows)

## Install from PyPI

The simplest way to install:

```bash
pip install isoext
```

This will compile the CUDA extension during installation.

```{note}
On Windows, you may encounter errors due to path length limits (260 characters).
Enable long paths by following [this guide](https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/).
```

## Install from Source

For development or to get the latest changes:

```bash
git clone https://github.com/GuangyanCai/isoext
cd isoext
pip install -e .
```

## Verify Installation

```python
import isoext

# Create a small test grid
grid = isoext.UniformGrid([8, 8, 8])
print(f"Grid has {grid.get_num_cells()} cells")

# Run marching cubes (should return empty mesh for default values)
v, f = isoext.marching_cubes(grid)
print(f"Extracted {len(f)} triangles")
```

## Troubleshooting

### CUDA not found

Make sure PyTorch is installed with CUDA support:

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.version.cuda)         # Should print your CUDA version
```

### Compilation errors

Ensure your CUDA toolkit version matches PyTorch's CUDA version. Check with:

```bash
nvcc --version
```

### Import errors

If you see `ImportError: PyTorch is required`, install PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version (e.g., `cu118`, `cu124`).

