"""Tests for the Vega (corrected MC33) marching cubes variant."""

import itertools

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


def extract_topology(values, method):
    grid = isoext.UniformGrid(list(values.shape))
    grid.set_values(values)
    v, f = isoext.marching_cubes(grid, method=method)
    return mesh_topology(len(v), f)


def random_closed_field(seed, size, sigma):
    """A smoothed random field whose surface does not cross the domain."""
    torch.manual_seed(seed)
    values = gaussian_smooth(torch.randn(size, size, size, device="cuda"), sigma)
    values[0], values[-1] = 1.0, 1.0
    values[:, 0], values[:, -1] = 1.0, 1.0
    values[:, :, 0], values[:, :, -1] = 1.0, 1.0
    return values


def cube_symmetries():
    """All 48 axis permutation and reflection combinations."""
    for perm in itertools.permutations(range(3)):
        for r in range(8):
            flips = [d for d in range(3) if r >> d & 1]
            yield perm, flips


def test_vega_sphere_matches_nagae(sphere_grid):
    """Spheres have no ambiguous cells, so vega must reproduce nagae."""
    v1, f1 = isoext.marching_cubes(sphere_grid, method="nagae")
    v2, f2 = isoext.marching_cubes(sphere_grid, method="vega")

    assert len(v1) == len(v2)
    assert len(f1) == len(f2)
    assert abs(signed_volume(v2, f2) - signed_volume(v1, f1)) < 1e-6


def test_vega_sphere_accuracy(sphere_grid):
    v, f = isoext.marching_cubes(sphere_grid, method="vega")

    err = (v.norm(dim=-1) - 0.5).abs()
    assert err.max().item() < (2.0 / 31) / 2


def test_vega_watertight_on_random_fields():
    """Random closed fields must produce meshes without boundary edges."""
    for seed in range(20):
        values = random_closed_field(seed, 16, sigma=1.5)
        grid = isoext.UniformGrid([16, 16, 16])
        grid.set_values(values)
        v, f = isoext.marching_cubes(grid, method="vega")
        if len(f) == 0:
            continue

        edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges = edges.sort(dim=-1).values
        _, counts = torch.unique(edges, dim=0, return_counts=True)
        assert (counts == 2).all(), f"open or non-manifold edges at seed {seed}"


def test_vega_symmetry_invariance():
    """The corrected interior test makes the extracted topology invariant
    under all 48 cube symmetries of the input field, which the Lewiner
    variant is not."""
    for seed in range(10):
        values = random_closed_field(seed, 9, sigma=1.0)
        reference = extract_topology(values, "vega")
        for perm, flips in cube_symmetries():
            transformed = values.permute(perm)
            if flips:
                transformed = transformed.flip(flips)
            topology = extract_topology(transformed.contiguous(), "vega")
            assert topology == reference, f"seed {seed}, permute {perm}, flip {flips}: {topology} != {reference}"


def test_vega_mirror_regression():
    """Single-cell fields where mirroring changes the Lewiner topology;
    vega must resolve both orientations the same. The second field is
    the one shown in the variants doc page; the first is exactly
    degenerate (the interpolant's neck has zero thickness)."""
    fields = [
        [0.0625, -0.0625, 0.1875, 0.0625, 0.0625, 0.0625, -0.0625, -0.0625],
        [0.5625, -0.4375, 0.0625, 0.5625, 0.1875, 0.125, -0.4375, 0.125],
    ]
    for flat in fields:
        values = torch.tensor(flat, device="cuda").reshape(2, 2, 2)
        mirrored = values.permute(2, 1, 0).contiguous()

        lewiner = [extract_topology(f, "lewiner") for f in (values, mirrored)]
        vega = [extract_topology(f, "vega") for f in (values, mirrored)]

        assert lewiner[0] != lewiner[1], f"expected Lewiner counterexample: {flat}"
        assert vega[0] == vega[1], flat


def test_vega_topology_matches_trilinear():
    """Compare against dense marching cubes on a trilinear upsampling of
    the field: inside each coarse cell the fine field samples the same
    trilinear interpolant whose topology MC33 promises to reproduce, and
    at 8x resolution the interpolant has no ambiguities left for the fine
    extraction to misjudge."""
    mismatches = []
    for seed in range(30):
        values = random_closed_field(seed, 5, sigma=0.8)
        coarse = extract_topology(values, "vega")

        fine_values = torch.nn.functional.interpolate(
            values[None, None], size=33, mode="trilinear", align_corners=True
        )[0, 0]
        fine = extract_topology(fine_values.contiguous(), "nagae")

        if coarse != fine:
            mismatches.append((seed, coarse, fine))

    assert not mismatches, mismatches


def test_vega_sparse_grid(sphere):
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    v, f = isoext.marching_cubes(grid, method="vega")

    assert len(v) > 0
    err = (v.norm(dim=-1) - 0.5).abs()
    assert err.max().item() < 2.0 / 31


def test_vega_surface_crossing_domain_boundary():
    sphere = SphereSDF(radius=1.2)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    grid.set_values(sphere(grid.get_points()))

    v, f = isoext.marching_cubes(grid, method="vega")

    assert len(v) > 0
    assert f.max().item() < len(v)
