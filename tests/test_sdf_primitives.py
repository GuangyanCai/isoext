"""Tests for SDF primitives (Sphere, Torus, Cuboid)."""

import torch

import isoext
from isoext.sdf import CuboidSDF, SphereSDF, TorusSDF


def test_sphere_sdf():
    """Test SphereSDF primitive."""
    sphere = SphereSDF(radius=0.5)
    points = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda")
    sdf_values = sphere(points)

    # Center should be -0.5 (inside)
    assert torch.isclose(sdf_values[0], torch.tensor(-0.5, device="cuda"))

    # On surface should be ~0
    assert torch.isclose(sdf_values[1], torch.tensor(0.0, device="cuda"), atol=1e-6)

    # Outside should be positive
    assert sdf_values[2] > 0


def test_torus_sdf():
    """Test TorusSDF primitive."""
    torus = TorusSDF(R=0.5, r=0.2)
    points = torch.tensor(
        [
            [0.5, 0.0, 0.0],  # On major radius, should be on surface
            [0.3, 0.0, 0.0],  # Inside the torus tube
            [1.0, 0.0, 0.0],  # Far outside
        ],
        device="cuda",
    )
    sdf_values = torus(points)

    # On major radius should be close to surface
    assert torch.abs(sdf_values[0]) < 0.3

    # Inside the torus tube should be negative
    assert sdf_values[1] < 0

    # Far outside should be positive
    assert sdf_values[2] > 0


def test_cuboid_sdf():
    """Test CuboidSDF primitive."""
    cuboid = CuboidSDF(size=[1.0, 1.0, 1.0])
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Center, should be inside
            [0.5, 0.0, 0.0],  # On surface
            [1.0, 0.0, 0.0],  # Outside
        ],
        device="cuda",
    )
    sdf_values = cuboid(points)

    # Center should be inside (negative)
    assert sdf_values[0] < 0

    # On surface should be ~0
    assert torch.abs(sdf_values[1]) < 1e-6

    # Outside should be positive
    assert sdf_values[2] > 0


def test_sphere_marching_cubes():
    """Test marching cubes extraction on sphere."""
    sphere = SphereSDF(radius=0.5)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3  # Vertices have 3 coordinates
    assert f.shape[1] == 3  # Faces have 3 vertices
    assert len(v) > 0  # Should have vertices
    assert len(f) > 0  # Should have faces


def test_torus_marching_cubes():
    """Test marching cubes extraction on torus."""
    torus = TorusSDF(R=0.5, r=0.2)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = torus(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_cuboid_marching_cubes():
    """Test marching cubes extraction on cuboid."""
    cuboid = CuboidSDF(size=[1.0, 1.0, 1.0])
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = cuboid(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0
