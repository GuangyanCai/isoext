"""Tests for dual marching cubes."""

import pytest
import torch

import isoext
from isoext.sdf import CuboidSDF, RotationOp, get_sdf_normal, project_to_surface

METHODS = ["vega", "lewiner", "nagae", "lorensen"]


def boundary_edge_count(f):
    edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = edges.sort(dim=-1).values
    _, counts = torch.unique(edges, dim=0, return_counts=True)
    return int((counts != 2).sum())


def make_sparse_sphere(shape, radius):
    """A sparse grid holding only the cells the surface crosses."""
    grid = isoext.SparseGrid([shape] * 3)
    for chunk in grid.get_potential_cell_indices(2_000_000):
        points = grid.get_points_by_cell_indices(chunk)
        near = grid.filter_cell_indices(chunk, points.norm(dim=-1) - radius)
        if len(near) > 0:
            grid.add_cells(near)
    grid.set_values(grid.get_points().norm(dim=-1) - radius)
    return grid


@pytest.mark.parametrize("method", METHODS)
def test_dmc_sphere(sphere_grid, method):
    v, f = isoext.dual_marching_cubes(sphere_grid, method=method)

    assert len(v) > 0
    assert torch.isfinite(v).all()
    assert boundary_edge_count(f) == 0
    err = (v.norm(dim=-1) - 0.5).abs()
    assert err.max().item() < (2.0 / 31) / 2
    # Outward orientation: positive enclosed volume.
    t = v[f.long()]
    volume = (torch.cross(t[:, 0], t[:, 1], dim=-1) * t[:, 2]).sum() / 6.0
    assert volume.item() > 0


def test_dmc_methods_agree_on_sphere(sphere_grid):
    """A sphere has no ambiguous dual cells, so all variants must produce
    the same mesh sizes."""
    sizes = set()
    for method in METHODS:
        v, f = isoext.dual_marching_cubes(sphere_grid, method=method)
        sizes.add((len(v), len(f)))
    assert len(sizes) == 1


def test_dmc_closed_on_random_fields():
    """The mesh is always closed: every edge bounds an even number of
    triangles. Where a tunnel passes through a cell face, the dual mesh
    has a non-manifold edge shared by exactly 4 triangles (a documented
    property of the method), but never a hole."""
    from isoext.utils import gaussian_smooth

    for seed in range(20):
        torch.manual_seed(seed)
        values = gaussian_smooth(torch.randn(16, 16, 16, device="cuda"), 1.5)
        # Two positive layers: the mesh retracts half a cell from the
        # boundary, so the surface must stay a full cell away for the
        # closedness guarantee to apply.
        values[:2], values[-2:] = 1.0, 1.0
        values[:, :2], values[:, -2:] = 1.0, 1.0
        values[:, :, :2], values[:, :, -2:] = 1.0, 1.0

        grid = isoext.UniformGrid([16, 16, 16])
        grid.set_values(values)
        v, f = isoext.dual_marching_cubes(grid)
        if len(f) == 0:
            continue
        assert torch.isfinite(v).all(), f"non-finite vertices at seed {seed}"
        edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges = edges.sort(dim=-1).values
        _, counts = torch.unique(edges, dim=0, return_counts=True)
        assert (counts != 1).all(), f"hole at seed {seed}"
        assert ((counts == 2) | (counts == 4)).all(), f"bad edge at seed {seed}"


def test_dmc_sharp_features():
    """With refined points and SDF normals, most vertices must lie exactly
    on the surface; the residual chords across feature lines stay small."""
    cube = RotationOp(sdf=CuboidSDF(size=[1.0, 1.0, 1.0]), axis=[1, 1, 0], angle=30)
    grid = isoext.UniformGrid([48] * 3)
    grid.set_values(cube(grid.get_points()))

    its = isoext.get_intersection(grid)
    points = project_to_surface(cube, its.get_points())
    its.set_points(points)
    its.set_normals(get_sdf_normal(cube, points))

    v, f = isoext.dual_marching_cubes(grid, intersection=its)
    d = cube(v).abs()
    cell = 2.0 / 47
    assert torch.quantile(d, 0.9).item() < 1e-4
    assert d.max().item() < cell


def test_dmc_sparse_grid():
    grid = make_sparse_sphere(32, 0.5)
    v, f = isoext.dual_marching_cubes(grid)

    assert len(v) > 0
    assert boundary_edge_count(f) == 0
    err = (v.norm(dim=-1) - 0.5).abs()
    assert err.max().item() < 2.0 / 31


def test_dmc_matches_dense_on_sparse():
    """A sparse grid holding only the crossed cells must reproduce the
    dense result away from the boundary."""
    dense = isoext.UniformGrid([32] * 3)
    dense.set_values(dense.get_points().norm(dim=-1) - 0.5)
    v1, f1 = isoext.dual_marching_cubes(dense)

    sparse = make_sparse_sphere(32, 0.5)
    v2, f2 = isoext.dual_marching_cubes(sparse)

    assert len(v1) == len(v2)
    assert len(f1) == len(f2)


def test_dmc_surface_crossing_domain_boundary():
    """The surface leaves the grid: the mesh is open but valid, retracted
    half a cell from the boundary."""
    grid = isoext.UniformGrid([32] * 3)
    grid.set_values(grid.get_points().norm(dim=-1) - 1.2)

    v, f = isoext.dual_marching_cubes(grid)

    assert len(v) > 0
    assert torch.isfinite(v).all()
    assert f.max().item() < len(v)


def test_dmc_tiny_sphere():
    """One inside sample on 2x2x2 cells: the dual of marching cubes'
    octahedron is a cube."""
    grid = isoext.UniformGrid([3, 3, 3])
    grid.set_values(grid.get_points().norm(dim=-1) - 0.7)

    v, f = isoext.dual_marching_cubes(grid)

    assert len(v) == 8
    assert len(f) == 12
    assert boundary_edge_count(f) == 0


def test_dmc_empty_field():
    grid = isoext.UniformGrid([16] * 3)
    grid.set_values(torch.ones(16, 16, 16, device="cuda"))

    v, f = isoext.dual_marching_cubes(grid)
    assert len(v) == 0
    assert len(f) == 0
