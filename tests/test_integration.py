"""Integration tests combining multiple features."""

import torch

import isoext
from isoext.sdf import (
    IntersectionOp,
    NegationOp,
    RotationOp,
    SphereSDF,
    TorusSDF,
    TranslationOp,
    UnionOp,
    get_sdf_normal,
)


def test_complex_shape_marching_cubes():
    """Test marching cubes on a complex composite shape."""
    # Create a sphere with three orthogonal toroidal holes
    torus_a = TorusSDF(R=0.75, r=0.15)
    torus_b = RotationOp(sdf=torus_a, axis=[1, 0, 0], angle=90)
    torus_c = RotationOp(sdf=torus_a, axis=[0, 1, 0], angle=90)
    sphere_a = SphereSDF(radius=0.75)
    sdf = IntersectionOp([sphere_a, NegationOp(UnionOp([torus_a, torus_b, torus_c]))])

    grid = isoext.UniformGrid([64, 64, 64], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_v = sdf(grid.get_points())
    grid.set_values(sdf_v)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_complex_shape_dual_contouring():
    """Test dual contouring on a complex composite shape."""
    torus_a = TorusSDF(R=0.75, r=0.15)
    torus_b = RotationOp(sdf=torus_a, axis=[1, 0, 0], angle=90)
    torus_c = RotationOp(sdf=torus_a, axis=[0, 1, 0], angle=90)
    sphere_a = SphereSDF(radius=0.75)
    sdf = IntersectionOp([sphere_a, NegationOp(UnionOp([torus_a, torus_b, torus_c]))])

    grid = isoext.UniformGrid([64, 64, 64], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_v = sdf(grid.get_points())
    grid.set_values(sdf_v)

    its = isoext.get_intersection(grid, level=0.0)
    points = its.get_points()

    if len(points) > 0:
        normals = get_sdf_normal(sdf, points)
        its.set_normals(normals)

        v, f = isoext.dual_contouring(grid, its, level=0.0)

        assert v.shape[1] == 3
        assert f.shape[1] == 3


def test_multiple_spheres_union():
    """Test union of multiple translated spheres."""
    spheres = [
        TranslationOp(SphereSDF(radius=0.3), offset=[-0.5, 0.0, 0.0]),
        TranslationOp(SphereSDF(radius=0.3), offset=[0.0, 0.0, 0.0]),
        TranslationOp(SphereSDF(radius=0.3), offset=[0.5, 0.0, 0.0]),
    ]
    union = UnionOp(spheres)

    grid = isoext.UniformGrid([48, 48, 48], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_v = union(grid.get_points())
    grid.set_values(sdf_v)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_nested_operations():
    """Test nested SDF operations."""
    sphere1 = SphereSDF(radius=0.4)
    sphere2 = TranslationOp(SphereSDF(radius=0.4), offset=[0.3, 0.0, 0.0])
    union = UnionOp([sphere1, sphere2])

    sphere3 = SphereSDF(radius=0.5)
    intersection = IntersectionOp([union, sphere3])

    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_v = intersection(grid.get_points())
    grid.set_values(sdf_v)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3


def test_rotated_translated_shape():
    """Test a shape that is both rotated and translated."""
    torus = TorusSDF(R=0.5, r=0.2)
    rotated = RotationOp(torus, axis=[1, 0, 0], angle=45)
    translated = TranslationOp(rotated, offset=[0.3, 0.0, 0.0])

    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_v = translated(grid.get_points())
    grid.set_values(sdf_v)

    v, f = isoext.marching_cubes(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
