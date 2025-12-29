"""Tests for SDF operations (Union, Intersection, Negation, Translation, Rotation, SmoothUnion)."""

import torch

import isoext
from isoext.sdf import (
    IntersectionOp,
    NegationOp,
    RotationOp,
    SmoothUnionOp,
    SphereSDF,
    TorusSDF,
    TranslationOp,
    UnionOp,
)


def test_union_op():
    """Test Union operation combining multiple SDFs."""
    sphere1 = SphereSDF(radius=0.3)
    sphere2 = SphereSDF(radius=0.3)
    union = UnionOp([sphere1, TranslationOp(sphere2, offset=[0.5, 0.0, 0.0])])

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Inside first sphere
            [0.5, 0.0, 0.0],  # Inside second sphere
            [0.25, 0.0, 0.0],  # In between, should be inside union
        ],
        device="cuda",
    )
    sdf_values = union(points)

    # All should be inside (negative)
    assert all(sdf_values < 0)


def test_intersection_op():
    """Test Intersection operation combining multiple SDFs."""
    sphere1 = SphereSDF(radius=0.5)
    sphere2 = SphereSDF(radius=0.5)
    intersection = IntersectionOp([sphere1, TranslationOp(sphere2, offset=[0.3, 0.0, 0.0])])

    points = torch.tensor(
        [
            [0.15, 0.0, 0.0],  # In intersection region
            [-0.4, 0.0, 0.0],  # Only in first sphere, outside second
            [0.7, 0.0, 0.0],  # Only in second sphere, outside first
        ],
        device="cuda",
    )
    sdf_values = intersection(points)

    # First point should be inside intersection (negative)
    assert sdf_values[0] < 0
    # Other points should be outside intersection (positive)
    assert sdf_values[1] > 0
    assert sdf_values[2] > 0


def test_negation_op():
    """Test Negation operation (inverts SDF)."""
    sphere = SphereSDF(radius=0.5)
    negated = NegationOp(sphere)

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Inside original sphere
            [1.0, 0.0, 0.0],  # Outside original sphere
        ],
        device="cuda",
    )
    sdf_values = negated(points)

    # Inside original should be outside negated (positive)
    assert sdf_values[0] > 0
    # Outside original should be inside negated (negative)
    assert sdf_values[1] < 0


def test_translation_op():
    """Test Translation operation."""
    sphere = SphereSDF(radius=0.3)
    translated = TranslationOp(sphere, offset=[0.5, 0.0, 0.0])

    points = torch.tensor(
        [
            [0.5, 0.0, 0.0],  # At translated center
            [0.0, 0.0, 0.0],  # At original center
        ],
        device="cuda",
    )
    sdf_values = translated(points)

    # At translated center should be inside
    assert sdf_values[0] < 0
    # At original center should be outside
    assert sdf_values[1] > 0


def test_rotation_op():
    """Test Rotation operation."""
    torus = TorusSDF(R=0.5, r=0.2)
    rotated = RotationOp(torus, axis=[1, 0, 0], angle=90)

    points = torch.tensor(
        [
            [0.0, 0.0, 0.5],  # Should be on rotated torus
            [0.5, 0.0, 0.0],  # Original torus position
        ],
        device="cuda",
    )
    sdf_values = rotated(points)

    # Should produce valid SDF values
    assert sdf_values.shape == (2,)


def test_rotation_op_radians():
    """Test Rotation operation with radians."""
    import math

    torus = TorusSDF(R=0.5, r=0.2)
    rotated = RotationOp(torus, axis=[1, 0, 0], angle=math.pi / 2, use_degree=False)

    points = torch.tensor([[0.0, 0.0, 0.5]], device="cuda")
    sdf_values = rotated(points)

    assert sdf_values.shape == (1,)


def test_smooth_union_op():
    """Test SmoothUnion operation."""
    sphere1 = SphereSDF(radius=0.3)
    sphere2 = SphereSDF(radius=0.3)
    smooth_union = SmoothUnionOp([sphere1, TranslationOp(sphere2, offset=[0.5, 0.0, 0.0])], k=0.1)

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        device="cuda",
    )
    sdf_values = smooth_union(points)

    # Should produce valid SDF values
    assert sdf_values.shape == (3,)
    # All should be inside (negative)
    assert all(sdf_values < 0)


def test_composite_shape_marching_cubes():
    """Test marching cubes on a composite shape."""
    sphere = SphereSDF(radius=0.5)
    torus = TorusSDF(R=0.4, r=0.15)
    union = UnionOp([sphere, torus])

    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = union(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_intersection_shape_marching_cubes():
    """Test marching cubes on an intersection shape."""
    sphere1 = SphereSDF(radius=0.5)
    sphere2 = SphereSDF(radius=0.5)
    intersection = IntersectionOp([sphere1, TranslationOp(sphere2, offset=[0.3, 0.0, 0.0])])

    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = intersection(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
