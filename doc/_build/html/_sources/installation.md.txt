# Installation

## Prerequisites

`isoext` requires:

- **Python** 3.10 or newer
- **PyTorch** with CUDA support
- **CUDA Toolkit** 12.4 or newer. Older nvcc versions fail on the glibc
  headers of recent Linux distributions (Ubuntu 24.04 and later) with
  errors like `"__builtin_dynamic_object_size" is undefined`. The build
  checks for this and stops with instructions if it finds an old nvcc.
- A C++ compiler (GCC on Linux, Visual Studio on Windows)

The three have to agree with each other: use the PyTorch build that
matches your CUDA toolkit's major version (compare `torch.version.cuda`
with `nvcc --version`), and a C++ compiler your CUDA toolkit supports --
each nvcc release accepts host compilers only up to a certain version.

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

To get the latest unreleased changes:

```bash
pip install git+https://github.com/GuangyanCai/isoext
```

For a development setup with the pinned toolchain, tests and docs, see
{doc}`development`.

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

### Slow first call

The extension ships GPU code as PTX. The driver compiles it for your GPU
the first time each algorithm runs, which takes a few seconds in total.
The result is cached on disk, so this happens once per machine.

### Import errors

If you see `ImportError: PyTorch is required`, install PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Replace `cu128` with your CUDA version.

