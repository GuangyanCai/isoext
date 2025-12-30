"""Tests for marching cubes algorithm with different methods."""

import torch

import isoext
from isoext.sdf import SphereSDF

from conftest import populate_sparse_grid


def test_marching_cubes_nagae(sphere_grid):
    """Test marching cubes with nagae method (default)."""
    v, f = isoext.marching_cubes(sphere_grid, level=0.0, method="nagae")

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_marching_cubes_lorensen(sphere_grid):
    """Test marching cubes with lorensen method."""
    v, f = isoext.marching_cubes(sphere_grid, level=0.0, method="lorensen")

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_marching_cubes_different_levels(sphere_grid):
    """Test marching cubes with different iso-levels."""
    # Test with positive level (smaller sphere)
    v1, f1 = isoext.marching_cubes(sphere_grid, level=0.1)
    assert v1.shape[1] == 3
    assert f1.shape[1] == 3

    # Test with negative level (larger sphere)
    v2, f2 = isoext.marching_cubes(sphere_grid, level=-0.1)
    assert v2.shape[1] == 3
    assert f2.shape[1] == 3

    # Both should produce valid meshes
    assert len(v1) > 0
    assert len(v2) > 0


def test_marching_cubes_different_resolutions(sphere):
    """Test marching cubes with different grid resolutions."""
    for res in [16, 32, 64]:
        grid = isoext.UniformGrid([res, res, res], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
        sdf_values = sphere(grid.get_points())
        grid.set_values(sdf_values)

        v, f = isoext.marching_cubes(grid, level=0.0)

        assert v.shape[1] == 3
        assert f.shape[1] == 3
        assert len(v) > 0
        assert len(f) > 0


def test_marching_cubes_empty_result():
    """Test marching cubes when no surface is found."""
    # Create a grid where all values are positive (no surface)
    grid = isoext.UniformGrid([8, 8, 8], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = torch.ones((8, 8, 8), device="cuda")
    grid.set_values(sdf_values)

    v, f = isoext.marching_cubes(grid, level=0.0)

    # Should return None or empty tensors when no surface is found
    if v is not None:
        assert v.shape[1] == 3
        assert f.shape[1] == 3


def test_marching_cubes_with_make_grid():
    """Test marching cubes using make_grid utility."""
    from isoext.utils import make_grid

    def sphere_sdf(x):
        return x.norm(dim=-1) - 0.5

    res = 16
    grid_tensor = make_grid([-1, -1, -1, 1, 1, 1], res=res, device="cuda")
    sdf = sphere_sdf(grid_tensor)

    # Create UniformGrid and set values
    grid = isoext.UniformGrid([res, res, res], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    grid.set_values(sdf)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_marching_cubes_non_uniform_resolution(sphere):
    """Test marching cubes with non-uniform resolution (different resolutions for each dimension)."""
    # Use different resolutions for x, y, z dimensions
    grid = isoext.UniformGrid([16, 32, 48], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    # Test with nagae method
    v1, f1 = isoext.marching_cubes(grid, level=0.0, method="nagae")
    assert v1.shape[1] == 3
    assert f1.shape[1] == 3
    assert len(v1) > 0
    assert len(f1) > 0

    # Test with lorensen method
    v2, f2 = isoext.marching_cubes(grid, level=0.0, method="lorensen")
    assert v2.shape[1] == 3
    assert f2.shape[1] == 3
    assert len(v2) > 0
    assert len(f2) > 0


def test_marching_cubes_non_uniform_resolution_extreme(sphere):
    """Test marching cubes with extreme non-uniform resolution ratios."""
    # Use very different resolutions (e.g., 8, 64, 16)
    grid = isoext.UniformGrid([8, 64, 16], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_marching_cubes_sparse_grid(sphere):
    """Test marching cubes with SparseGrid."""
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_marching_cubes_sparse_grid_lorensen(sphere):
    """Test marching cubes with SparseGrid using lorensen method."""
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    v, f = isoext.marching_cubes(grid, level=0.0, method="lorensen")

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_marching_cubes_sparse_grid_different_levels(sphere):
    """Test marching cubes with SparseGrid at different iso-levels."""
    shape = [32, 32, 32]

    for level in [-0.1, 0.0, 0.1]:
        grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
        populate_sparse_grid(grid, sphere, shape, level=level)

        if grid.get_num_cells() > 0:
            v, f = isoext.marching_cubes(grid, level=level)

            assert v.shape[1] == 3
            assert f.shape[1] == 3
