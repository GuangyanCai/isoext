"""Tests for the marching tetrahedra algorithm."""

import torch

import isoext
from isoext.sdf import SphereSDF

from conftest import populate_sparse_grid


def test_marching_tetrahedra_simple(sphere_grid):
    v, f = isoext.marching_tetrahedra(sphere_grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_marching_tetrahedra_vertex_accuracy(sphere_grid):
    """Vertices interpolated on tetrahedron edges must lie near the surface."""
    v, f = isoext.marching_tetrahedra(sphere_grid, level=0.0)

    err = (v.norm(dim=-1) - 0.5).abs()
    cell_size = 2.0 / 31
    assert err.max().item() < cell_size / 2


def test_marching_tetrahedra_triangle_count(sphere_grid):
    """MT produces roughly 2-3x the triangles of marching cubes."""
    _, f_mc = isoext.marching_cubes(sphere_grid)
    _, f_mt = isoext.marching_tetrahedra(sphere_grid)

    ratio = len(f_mt) / len(f_mc)
    assert 1.5 < ratio < 4.0


def test_marching_tetrahedra_watertight(sphere_grid):
    """A closed surface must produce a closed mesh: V - E + F == 2."""
    v, f = isoext.marching_tetrahedra(sphere_grid, level=0.0)

    edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = edges.sort(dim=-1).values
    unique_edges = torch.unique(edges, dim=0)

    euler = len(v) - len(unique_edges) + len(f)
    assert euler == 2


def test_marching_tetrahedra_winding_matches_marching_cubes(sphere_grid):
    """Signed volumes must agree in sign and value with marching cubes."""

    def signed_volume(v, f):
        tri = v[f.long()]
        return (
            torch.cross(tri[:, 0], tri[:, 1], dim=-1) * tri[:, 2]
        ).sum() / 6.0

    v_mc, f_mc = isoext.marching_cubes(sphere_grid)
    v_mt, f_mt = isoext.marching_tetrahedra(sphere_grid)

    vol_mc = signed_volume(v_mc, f_mc).item()
    vol_mt = signed_volume(v_mt, f_mt).item()

    assert vol_mc * vol_mt > 0
    assert abs(vol_mt - vol_mc) / abs(vol_mc) < 0.05


def test_marching_tetrahedra_empty_result():
    grid = isoext.UniformGrid([8, 8, 8], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    grid.set_values(torch.ones((8, 8, 8), device="cuda"))

    v, f = isoext.marching_tetrahedra(grid)

    assert isinstance(v, torch.Tensor)
    assert isinstance(f, torch.Tensor)
    assert v.shape == (0, 3)
    assert f.shape == (0, 3)


def test_marching_tetrahedra_surface_crossing_domain_boundary():
    sphere = SphereSDF(radius=1.2)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    grid.set_values(sphere(grid.get_points()))

    v, f = isoext.marching_tetrahedra(grid, level=0.0)

    assert len(v) > 0
    assert f.max().item() < len(v)


def test_marching_tetrahedra_sparse_grid(sphere):
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    v, f = isoext.marching_tetrahedra(grid, level=0.0)

    assert len(v) > 0
    err = (v.norm(dim=-1) - 0.5).abs()
    assert err.max().item() < 2.0 / 31
