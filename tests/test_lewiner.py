"""Tests for the Lewiner (MC33) marching cubes variant."""

import pytest
import torch
from conftest import populate_sparse_grid

import isoext
from isoext.sdf import SphereSDF
from isoext.utils import gaussian_smooth


def signed_volume(v, f):
    tri = v[f.long()]
    return (torch.cross(tri[:, 0], tri[:, 1], dim=-1) * tri[:, 2]).sum().item() / 6.0


def mesh_topology(num_vertices, faces):
    """Return (euler characteristic, connected components) of a mesh."""
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = edges.sort(dim=-1).values
    num_edges = len(torch.unique(edges, dim=0))
    euler = num_vertices - num_edges + len(faces)

    parent = list(range(num_vertices))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    used = set()
    for a, b, c in faces.cpu().tolist():
        used.update((a, b, c))
        parent[find(b)] = find(a)
        parent[find(c)] = find(a)
    components = len({find(a) for a in used})
    return euler, components


def test_lewiner_sphere_matches_nagae(sphere_grid):
    """Spheres have no ambiguous cells, so mc33 must reproduce nagae."""
    v1, f1 = isoext.marching_cubes(sphere_grid, method="nagae")
    v2, f2 = isoext.marching_cubes(sphere_grid, method="lewiner")

    assert len(v1) == len(v2)
    assert len(f1) == len(f2)
    assert abs(signed_volume(v2, f2) - signed_volume(v1, f1)) < 1e-6


def test_lewiner_sphere_accuracy(sphere_grid):
    v, f = isoext.marching_cubes(sphere_grid, method="lewiner")

    err = (v.norm(dim=-1) - 0.5).abs()
    assert err.max().item() < (2.0 / 31) / 2


def test_lewiner_watertight_on_random_fields():
    """Random closed fields must produce meshes without boundary edges."""
    for seed in range(20):
        torch.manual_seed(seed)
        values = gaussian_smooth(torch.randn(16, 16, 16, device="cuda"), sigma=1.5)
        values[0], values[-1] = 1.0, 1.0
        values[:, 0], values[:, -1] = 1.0, 1.0
        values[:, :, 0], values[:, :, -1] = 1.0, 1.0

        grid = isoext.UniformGrid([16, 16, 16])
        grid.set_values(values)
        v, f = isoext.marching_cubes(grid, method="lewiner")
        if len(f) == 0:
            continue

        edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges = edges.sort(dim=-1).values
        _, counts = torch.unique(edges, dim=0, return_counts=True)
        assert (counts == 2).all(), f"open or non-manifold edges at seed {seed}"


def test_lewiner_topology_matches_skimage():
    """Compare topology against scikit-image's Lewiner implementation."""
    skimage_measure = pytest.importorskip("skimage.measure")

    mismatches = []
    for seed in range(100):
        torch.manual_seed(seed)
        values = gaussian_smooth(torch.randn(9, 9, 9, device="cuda"), sigma=1.0)
        values[0], values[-1] = 1.0, 1.0
        values[:, 0], values[:, -1] = 1.0, 1.0
        values[:, :, 0], values[:, :, -1] = 1.0, 1.0

        grid = isoext.UniformGrid([9, 9, 9])
        grid.set_values(values)
        v_ours, f_ours = isoext.marching_cubes(grid, method="lewiner")

        # Transpose so scikit-image processes identically oriented cells:
        # its first array axis is z. Lewiner's simplified interior test is
        # not orientation-invariant, so mismatched orientations would
        # produce spurious topology differences.
        volume = values.permute(2, 1, 0).cpu().numpy().copy()
        try:
            v_ref, f_ref, _, _ = skimage_measure.marching_cubes(volume, level=0.0)
        except (ValueError, RuntimeError):
            assert len(f_ours) == 0
            continue

        ours = mesh_topology(len(v_ours), f_ours)
        ref = mesh_topology(len(v_ref), torch.as_tensor(f_ref.astype(int).copy()))
        if ours != ref:
            mismatches.append((seed, ours, ref))

    assert not mismatches, mismatches


def test_lewiner_sparse_grid(sphere):
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    v, f = isoext.marching_cubes(grid, method="lewiner")

    assert len(v) > 0
    err = (v.norm(dim=-1) - 0.5).abs()
    assert err.max().item() < 2.0 / 31


def test_lewiner_surface_crossing_domain_boundary():
    sphere = SphereSDF(radius=1.2)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    grid.set_values(sphere(grid.get_points()))

    v, f = isoext.marching_cubes(grid, method="lewiner")

    assert len(v) > 0
    assert f.max().item() < len(v)
