"""Tests for the surface nets algorithm."""

import torch
from conftest import populate_sparse_grid

import isoext
from isoext.sdf import SphereSDF


def test_surface_nets_simple(sphere_grid):
    v, f = isoext.surface_nets(sphere_grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_surface_nets_vertex_accuracy(sphere_grid):
    """Centroid vertices must lie close to the sphere surface."""
    v, f = isoext.surface_nets(sphere_grid, level=0.0)

    err = (v.norm(dim=-1) - 0.5).abs()
    cell_size = 2.0 / 31
    assert err.max().item() < cell_size


def test_surface_nets_with_intersection(sphere_grid):
    its = isoext.get_intersection(sphere_grid, level=0.0)

    v, f = isoext.surface_nets(sphere_grid, level=0.0, intersection=its)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0


def test_surface_nets_empty_result():
    grid = isoext.UniformGrid([8, 8, 8], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    grid.set_values(torch.ones((8, 8, 8), device="cuda"))

    v, f = isoext.surface_nets(grid, level=0.0)

    assert isinstance(v, torch.Tensor)
    assert isinstance(f, torch.Tensor)
    assert v.shape == (0, 3)
    assert f.shape == (0, 3)


def test_surface_nets_surface_crossing_domain_boundary():
    """Boundary edges must be skipped, like in dual contouring."""
    sphere = SphereSDF(radius=1.2)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    grid.set_values(sphere(grid.get_points()))

    v, f = isoext.surface_nets(grid, level=0.0)

    assert len(v) > 0
    assert f.max().item() < len(v)

    tri = v[f.long()]
    edge_len = torch.cat(
        [
            (tri[:, 0] - tri[:, 1]).norm(dim=-1),
            (tri[:, 1] - tri[:, 2]).norm(dim=-1),
            (tri[:, 2] - tri[:, 0]).norm(dim=-1),
        ]
    )
    cell_size = 2.0 / 31
    assert edge_len.max().item() < 4 * cell_size


def test_surface_nets_sparse_grid(sphere):
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    v, f = isoext.surface_nets(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    err = (v.norm(dim=-1) - 0.5).abs()
    assert err.max().item() < 2.0 / 31
