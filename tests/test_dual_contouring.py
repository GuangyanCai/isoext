"""Tests for dual contouring algorithm."""

import torch

import isoext
from isoext.sdf import SphereSDF, get_sdf_normal

from conftest import populate_sparse_grid


def test_dual_contouring_simple(sphere_grid):
    """Test simplest dual contouring usage - no intersection provided."""
    v, f = isoext.dual_contouring(sphere_grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_with_intersection_auto_normals(sphere_grid):
    """Test dual contouring with intersection but normals computed automatically."""
    # Get intersection without computing normals
    its = isoext.get_intersection(sphere_grid, level=0.0, compute_normals=False)
    assert not its.has_normals()

    # Dual contouring should compute normals automatically
    v, f = isoext.dual_contouring(sphere_grid, level=0.0, intersection=its)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_with_custom_normals(sphere, sphere_grid):
    """Test dual contouring with user-provided normals."""
    # Get intersection points
    its = isoext.get_intersection(sphere_grid, level=0.0)
    points = its.get_points()

    # Compute custom normals using SDF gradient
    normals = get_sdf_normal(sphere, points)
    its.set_normals(normals)
    assert its.has_normals()

    # Run dual contouring with custom normals
    v, f = isoext.dual_contouring(sphere_grid, level=0.0, intersection=its)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_different_levels(sphere_grid):
    """Test dual contouring with different iso-levels."""
    for level in [-0.1, 0.0, 0.1]:
        v, f = isoext.dual_contouring(sphere_grid, level=level)
        if len(v) > 0:
            assert v.shape[1] == 3
            assert f.shape[1] == 3


def test_dual_contouring_parameters(sphere_grid):
    """Test dual contouring with different regularization parameters."""
    v, f = isoext.dual_contouring(sphere_grid, level=0.0, reg=0.01)

    assert v.shape[1] == 3
    assert f.shape[1] == 3


def test_get_intersection(sphere_grid):
    """Test getting intersection points from grid."""
    its = isoext.get_intersection(sphere_grid, level=0.0)
    points = its.get_points()

    assert points.shape[1] == 3
    assert len(points) > 0
    assert not its.has_normals()  # Normals not computed by default


def test_get_intersection_with_normals(sphere_grid):
    """Test getting intersection with normals computed."""
    its = isoext.get_intersection(sphere_grid, level=0.0, compute_normals=True)
    points = its.get_points()
    normals = its.get_normals()

    assert points.shape[1] == 3
    assert normals.shape[1] == 3
    assert len(points) > 0
    assert its.has_normals()


def test_intersection_set_normals(sphere, sphere_grid):
    """Test setting normals on intersection object."""
    its = isoext.get_intersection(sphere_grid, level=0.0)
    assert not its.has_normals()

    points = its.get_points()
    if len(points) > 0:
        normals = get_sdf_normal(sphere, points)
        its.set_normals(normals)
        assert its.has_normals()

        # Verify normals were set
        retrieved_normals = its.get_normals()
        assert retrieved_normals.shape == normals.shape
        assert torch.allclose(retrieved_normals, normals, atol=1e-5)


def test_dual_contouring_non_uniform_resolution(sphere):
    """Test dual contouring with non-uniform resolution."""
    grid = isoext.UniformGrid([16, 32, 48], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.dual_contouring(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_non_uniform_resolution_extreme(sphere):
    """Test dual contouring with extreme non-uniform resolution ratios."""
    grid = isoext.UniformGrid([8, 64, 16], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.dual_contouring(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_get_intersection_non_uniform_resolution(sphere):
    """Test getting intersection points from non-uniform resolution grid."""
    grid = isoext.UniformGrid([16, 32, 48], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    its = isoext.get_intersection(grid, level=0.0, compute_normals=True)
    points = its.get_points()
    normals = its.get_normals()

    assert points.shape[1] == 3
    assert normals.shape[1] == 3
    assert its.has_normals()
    assert len(points) > 0


def test_dual_contouring_sparse_grid(sphere):
    """Test dual contouring with SparseGrid."""
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    v, f = isoext.dual_contouring(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_sparse_grid_with_custom_normals(sphere):
    """Test dual contouring with SparseGrid and custom normals."""
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    # Get intersection and compute custom normals
    its = isoext.get_intersection(grid, level=0.0)
    its_points = its.get_points()
    normals = get_sdf_normal(sphere, its_points)
    its.set_normals(normals)
    assert its.has_normals()

    v, f = isoext.dual_contouring(grid, level=0.0, intersection=its)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_get_intersection_sparse_grid(sphere):
    """Test getting intersection points from SparseGrid."""
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    its = isoext.get_intersection(grid, level=0.0, compute_normals=True)
    its_points = its.get_points()
    its_normals = its.get_normals()

    assert its_points.shape[1] == 3
    assert its_normals.shape[1] == 3
    assert its.has_normals()
    assert len(its_points) > 0


def test_dual_contouring_sparse_grid_different_levels(sphere):
    """Test dual contouring with SparseGrid at different iso-levels."""
    shape = [32, 32, 32]

    for level in [-0.1, 0.0, 0.1]:
        grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
        populate_sparse_grid(grid, sphere, shape, level=level)

        if grid.get_num_cells() > 0:
            v, f = isoext.dual_contouring(grid, level=level)

            assert v.shape[1] == 3
            assert f.shape[1] == 3
