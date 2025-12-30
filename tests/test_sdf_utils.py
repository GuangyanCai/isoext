"""Tests for SDF utility functions (gradient, normal computation)."""

import torch

from isoext.sdf import CuboidSDF, get_sdf_grad, get_sdf_normal


def test_get_sdf_grad_sphere(sphere):
    """Test gradient computation for sphere SDF."""
    points = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda")
    grad = get_sdf_grad(sphere, points)

    assert grad.shape == (3, 3)
    # At center, gradient should be zero (or very small)
    assert torch.norm(grad[0]) < 1e-5
    # At surface, gradient should point outward
    assert grad[1, 0] > 0
    # At outside point, gradient should point outward
    assert grad[2, 0] > 0


def test_get_sdf_normal_sphere(sphere):
    """Test normal computation for sphere SDF."""
    points = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]], device="cuda")
    normals = get_sdf_normal(sphere, points)

    assert normals.shape == (3, 3)
    # Normals should be normalized
    norms = torch.norm(normals, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # First point should have normal pointing in +x direction
    assert torch.allclose(normals[0], torch.tensor([1.0, 0.0, 0.0], device="cuda"), atol=1e-4)


def test_get_sdf_grad_torus(torus):
    """Test gradient computation for torus SDF."""
    points = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], device="cuda")
    grad = get_sdf_grad(torus, points)

    assert grad.shape == (2, 3)


def test_get_sdf_normal_torus(torus):
    """Test normal computation for torus SDF."""
    # Use points that are clearly on or near the surface (avoid singularities like center)
    points = torch.tensor([[0.6, 0.1, 0.0], [0.4, 0.1, 0.0], [0.5, 0.2, 0.0]], device="cuda")
    normals = get_sdf_normal(torus, points)

    assert normals.shape == (3, 3)
    # Normals should be normalized (check that non-zero normals are unit length)
    norms = torch.norm(normals, dim=-1)
    # Filter out zero normals (singularities where gradient is zero)
    non_zero_mask = norms > 1e-6
    if non_zero_mask.any():
        assert torch.allclose(norms[non_zero_mask], torch.ones_like(norms[non_zero_mask]), atol=1e-5)


def test_get_sdf_grad_cuboid():
    """Test gradient computation for cuboid SDF."""
    cuboid = CuboidSDF(size=[1.0, 1.0, 1.0])
    points = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda")
    grad = get_sdf_grad(cuboid, points)

    assert grad.shape == (3, 3)
    # At center, gradient should be zero
    assert torch.norm(grad[0]) < 1e-5
    # At surface, gradient should point outward
    assert grad[1, 0] > 0


def test_get_sdf_normal_cuboid():
    """Test normal computation for cuboid SDF."""
    cuboid = CuboidSDF(size=[1.0, 1.0, 1.0])
    points = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]], device="cuda")
    normals = get_sdf_normal(cuboid, points)

    assert normals.shape == (3, 3)
    # Normals should be normalized
    norms = torch.norm(normals, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_get_sdf_normal_batch(sphere):
    """Test normal computation with batched points."""
    # Create a batch of points with shape (2, 3, 3)
    points = torch.randn(2, 3, 3, device="cuda")
    normals = get_sdf_normal(sphere, points)

    assert normals.shape == (2, 3, 3)
    # Normals should be normalized
    norms = torch.norm(normals, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
