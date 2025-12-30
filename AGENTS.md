# Agent Guide for isoext

This document helps AI agents navigate and work with the isoext codebase.

## Overview

isoext is a GPU-accelerated library for iso-surface extraction (marching cubes, dual contouring) with PyTorch integration. It uses:
- **CUDA/C++** for the core algorithms
- **nanobind** for Python bindings
- **pixi** for environment and task management

## Project Structure

```
isoext/
├── src/
│   ├── isoext/           # Python package (sdf.py, utils.py, __init__.py)
│   ├── isoext_ext.cu     # Python bindings (nanobind)
│   ├── mc/               # Marching cubes implementations
│   ├── grid/             # Grid implementations (uniform, sparse)
│   ├── dc.cu             # Dual contouring
│   └── its.cu            # Intersection computation
├── include/              # CUDA headers (.cuh files)
├── tests/                # pytest tests
│   └── conftest.py       # Shared fixtures
├── doc/                  # Sphinx documentation
├── luts/                 # Lookup table generation
├── pyproject.toml        # Project config and pixi tasks
└── CMakeLists.txt        # Build configuration
```

## Running Commands

This project uses **pixi** for environment management. Always use pixi to run commands.

### Environments

- `cu128` - Main development environment with PyTorch + CUDA 12.8
- `dev` - Development tools only (no PyTorch)
- `doc` - Documentation building

### Common Tasks

```bash
# Compile the extension (auto-rebuilds on source changes)
pixi run --environment cu128 compile

# Run all tests (compiles first)
pixi run --environment cu128 test

# Run specific test file
pixi run --environment cu128 pytest tests/test_marching_cubes.py -v

# Build documentation
pixi run --environment doc doc-build

# Serve docs with auto-reload
pixi run --environment doc doc-serve
```

**Important**: The `test` task depends on `compile`, so it will rebuild automatically. Use `--environment cu128` for any task requiring PyTorch/CUDA.

## Making Changes

### Python Code

Python source is in `src/isoext/`:
- `sdf.py` - SDF primitives and operations
- `utils.py` - Utilities (make_grid, write_obj)
- `__init__.py` - Package exports

Changes take effect immediately (no rebuild needed).

### C++/CUDA Code

Source files are in `src/` and headers in `include/`. After editing:
1. Run `pixi run --environment cu128 compile` (or just run tests, which compiles first)
2. The build system uses ninja and is incremental

Key files:
- `src/isoext_ext.cu` - Python bindings (nanobind)
- `src/mc/` - Marching cubes variants
- `src/dc.cu` - Dual contouring
- `src/its.cu` - Intersection/normal computation
- `src/grid/` - UniformGrid and SparseGrid

### Tests

Tests are in `tests/` using pytest:
- `conftest.py` - Shared fixtures (`sphere`, `sphere_grid`, `torus`, `torus_grid`, `populate_sparse_grid`)
- Test files follow `test_*.py` naming

To add tests, use fixtures from conftest.py to avoid boilerplate:
```python
def test_something(sphere_grid):
    v, f = isoext.marching_cubes(sphere_grid)
    assert v.shape[1] == 3
```

## Code Patterns

### Grid Setup Pattern

```python
import isoext
from isoext.sdf import SphereSDF

sphere = SphereSDF(radius=0.5)
grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1,-1,-1], aabb_max=[1,1,1])
grid.set_values(sphere(grid.get_points()))
```

### SparseGrid Pattern

Use `populate_sparse_grid` from conftest or follow this pattern:
```python
shape = [32, 32, 32]
grid = isoext.SparseGrid(shape, aabb_min=[-1,-1,-1], aabb_max=[1,1,1])
chunks = grid.get_potential_cell_indices(shape[0] * shape[1] * shape[2])
for chunk in chunks:
    points = grid.get_points_by_cell_indices(chunk)
    sdf_values = sdf(points)
    filtered = grid.filter_cell_indices(chunk, sdf_values, level=0.0)
    if len(filtered) > 0:
        grid.add_cells(filtered)
# Then set values for active cells
grid.set_values(sdf(grid.get_points()))
```

## Debugging Tips

1. **Build errors**: Check CUDA toolkit is available (`nvcc --version`)
2. **Import errors**: Make sure you compiled first (`pixi run --environment cu128 compile`)
3. **Test failures**: Run specific test with `-v` for verbose output
4. **Type stubs**: Auto-generated at `src/isoext/isoext_ext.pyi` during build

## Documentation

Documentation uses Sphinx with MyST-NB (Jupyter notebooks):
- Source: `doc/`
- Build output: `doc/_build/html/`
- Notebooks are executed during build

To preview docs while editing:
```bash
pixi run --environment doc doc-serve
```

