"""Tests for marching cubes algorithm with different methods."""

import torch

import isoext
from isoext.sdf import SphereSDF


def test_marching_cubes_nagae():
    """Test marching cubes with nagae method (default)."""
    sphere = SphereSDF(radius=0.5)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.marching_cubes(grid, level=0.0, method="nagae")

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_marching_cubes_lorensen():
    """Test marching cubes with lorensen method."""
    sphere = SphereSDF(radius=0.5)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.marching_cubes(grid, level=0.0, method="lorensen")

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_marching_cubes_different_levels():
    """Test marching cubes with different iso-levels."""
    sphere = SphereSDF(radius=0.5)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    # Test with positive level (smaller sphere)
    v1, f1 = isoext.marching_cubes(grid, level=0.1)
    assert v1.shape[1] == 3
    assert f1.shape[1] == 3

    # Test with negative level (larger sphere)
    v2, f2 = isoext.marching_cubes(grid, level=-0.1)
    assert v2.shape[1] == 3
    assert f2.shape[1] == 3

    # Both should produce valid meshes
    assert len(v1) > 0
    assert len(v2) > 0


def test_marching_cubes_different_resolutions():
    """Test marching cubes with different grid resolutions."""
    sphere = SphereSDF(radius=0.5)

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


def test_marching_cubes_non_uniform_resolution():
    """Test marching cubes with non-uniform resolution (different resolutions for each dimension)."""
    sphere = SphereSDF(radius=0.5)
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


def test_marching_cubes_non_uniform_resolution_extreme():
    """Test marching cubes with extreme non-uniform resolution ratios."""
    sphere = SphereSDF(radius=0.5)
    # Use very different resolutions (e.g., 8, 64, 16)
    grid = isoext.UniformGrid([8, 64, 16], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0
