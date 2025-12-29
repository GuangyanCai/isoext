"""Tests for dual contouring algorithm."""

import torch

import isoext
from isoext.sdf import SphereSDF, get_sdf_normal


def test_dual_contouring_basic():
    """Test basic dual contouring extraction."""
    sphere = SphereSDF(radius=0.5)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    # Get intersection points
    its = isoext.get_intersection(grid, level=0.0)
    points = its.get_points()

    # Compute normals
    normals = get_sdf_normal(sphere, points)
    its.set_normals(normals)

    # Run dual contouring
    v, f = isoext.dual_contouring(grid, its, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_different_levels():
    """Test dual contouring with different iso-levels."""
    sphere = SphereSDF(radius=0.5)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    for level in [-0.1, 0.0, 0.1]:
        its = isoext.get_intersection(grid, level=level)
        points = its.get_points()

        if len(points) > 0:
            normals = get_sdf_normal(sphere, points)
            its.set_normals(normals)

            v, f = isoext.dual_contouring(grid, its, level=level)

            assert v.shape[1] == 3
            assert f.shape[1] == 3


def test_dual_contouring_parameters():
    """Test dual contouring with different regularization parameters."""
    sphere = SphereSDF(radius=0.5)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    its = isoext.get_intersection(grid, level=0.0)
    points = its.get_points()

    if len(points) > 0:
        normals = get_sdf_normal(sphere, points)
        its.set_normals(normals)

        # Test with different regularization
        v, f = isoext.dual_contouring(grid, its, level=0.0, reg=0.01)

        assert v.shape[1] == 3
        assert f.shape[1] == 3


def test_get_intersection():
    """Test getting intersection points from grid."""
    sphere = SphereSDF(radius=0.5)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    its = isoext.get_intersection(grid, level=0.0)
    points = its.get_points()
    normals = its.get_normals()

    assert points.shape[1] == 3
    assert normals.shape[1] == 3
    assert len(points) > 0


def test_intersection_set_normals():
    """Test setting normals on intersection object."""
    sphere = SphereSDF(radius=0.5)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    its = isoext.get_intersection(grid, level=0.0)
    points = its.get_points()

    if len(points) > 0:
        normals = get_sdf_normal(sphere, points)
        its.set_normals(normals)

        # Verify normals were set
        retrieved_normals = its.get_normals()
        assert retrieved_normals.shape == normals.shape
        assert torch.allclose(retrieved_normals, normals, atol=1e-5)


def test_dual_contouring_non_uniform_resolution():
    """Test dual contouring with non-uniform resolution (different resolutions for each dimension)."""
    sphere = SphereSDF(radius=0.5)
    # Use different resolutions for x, y, z dimensions
    grid = isoext.UniformGrid([16, 32, 48], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    # Get intersection points
    its = isoext.get_intersection(grid, level=0.0)
    points = its.get_points()

    if len(points) > 0:
        # Compute normals
        normals = get_sdf_normal(sphere, points)
        its.set_normals(normals)

        # Run dual contouring
        v, f = isoext.dual_contouring(grid, its, level=0.0)

        assert v.shape[1] == 3
        assert f.shape[1] == 3
        assert len(v) > 0
        assert len(f) > 0


def test_dual_contouring_non_uniform_resolution_extreme():
    """Test dual contouring with extreme non-uniform resolution ratios."""
    sphere = SphereSDF(radius=0.5)
    # Use very different resolutions (e.g., 8, 64, 16)
    grid = isoext.UniformGrid([8, 64, 16], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    # Get intersection points
    its = isoext.get_intersection(grid, level=0.0)
    points = its.get_points()

    if len(points) > 0:
        # Compute normals
        normals = get_sdf_normal(sphere, points)
        its.set_normals(normals)

        # Run dual contouring
        v, f = isoext.dual_contouring(grid, its, level=0.0)

        assert v.shape[1] == 3
        assert f.shape[1] == 3
        assert len(v) > 0
        assert len(f) > 0


def test_get_intersection_non_uniform_resolution():
    """Test getting intersection points from non-uniform resolution grid."""
    sphere = SphereSDF(radius=0.5)
    grid = isoext.UniformGrid([16, 32, 48], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    its = isoext.get_intersection(grid, level=0.0)
    points = its.get_points()
    normals = its.get_normals()

    assert points.shape[1] == 3
    assert normals.shape[1] == 3
    if len(points) > 0:
        assert len(points) > 0
