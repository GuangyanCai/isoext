"""Tests for the mesh SDF (distance to a triangle mesh on a GPU BVH)."""

import pytest
import torch

import isoext
from isoext.sdf import CuboidSDF, RotationOp, SphereSDF, TriangleMeshSDF, get_sdf_normal, project_to_surface


def extract(sdf, n=64):
    grid = isoext.UniformGrid([n, n, n])
    grid.set_values(sdf(grid.get_points()))
    return isoext.marching_cubes(grid), 2.0 / (n - 1)


def random_points(n, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(n, 3, device="cuda", generator=gen) * 1.8 - 0.9


@pytest.mark.parametrize(
    "sdf", [SphereSDF(radius=0.5), RotationOp(sdf=CuboidSDF(size=[1.0, 1.0, 1.0]), axis=[1, 1, 0], angle=30)]
)
def test_mesh_sdf_matches_analytic_sdf(sdf):
    """The distance to an extracted mesh must match the analytic SDF within the mesh's own error."""
    (v, f), cell = extract(sdf)
    mesh_sdf = TriangleMeshSDF(v, f)

    p = random_points(20000)
    err = (mesh_sdf(p) - sdf(p)).abs()
    # Marching cubes vertices sit on the interpolated surface, within a
    # fraction of a cell of the true one; at the cube's edges the mesh is
    # chamfered by up to a cell.
    assert err.max().item() < cell
    assert err.mean().item() < 0.1 * cell


def test_mesh_sdf_sign():
    """Inside is negative, outside positive, for a closed mesh."""
    (v, f), _ = extract(SphereSDF(radius=0.5))
    mesh_sdf = TriangleMeshSDF(v, f)
    p = random_points(20000, seed=1)
    inside = p.norm(dim=-1) < 0.45
    outside = p.norm(dim=-1) > 0.55
    d = mesh_sdf(p)
    assert (d[inside] < 0).all()
    assert (d[outside] > 0).all()


def test_mesh_sdf_unsigned():
    """signed=False returns the unsigned distance, also for an open mesh."""
    (v, f), _ = extract(SphereSDF(radius=0.5))
    # Keep only the upper half of the triangles: an open bowl.
    keep = v[f.long()][:, :, 2].mean(dim=-1) > 0
    bowl = TriangleMeshSDF(v, f[keep], signed=False)
    p = random_points(5000, seed=2)
    assert (bowl(p) >= 0).all()
    # Points on the upper half of the sphere are as close to the bowl as to
    # the full sphere.
    upper = p[p[:, 2] > 0.3]
    full = TriangleMeshSDF(v, f, signed=False)
    assert torch.allclose(bowl(upper), full(upper), atol=1e-6)


def test_mesh_sdf_gradient_and_projection():
    """The gradient is the unit direction away from the closest point, so
    project_to_surface lands on the mesh."""
    (v, f), cell = extract(SphereSDF(radius=0.5))
    mesh_sdf = TriangleMeshSDF(v, f)
    p = random_points(5000, seed=3)
    p = p[(p.norm(dim=-1) - 0.5).abs() > 0.05]  # away from the surface

    n = get_sdf_normal(mesh_sdf, p)
    expected = torch.nn.functional.normalize(p, dim=-1)
    # The mesh normals differ from the sphere's by the faceting angle.
    assert (n - expected).norm(dim=-1).max().item() < 0.2
    assert (n.norm(dim=-1) - 1).abs().max().item() < 1e-5

    q = project_to_surface(mesh_sdf, p)
    assert mesh_sdf(q).abs().max().item() < 1e-4 * cell + 1e-5


def test_mesh_sdf_closest_points():
    (v, f), _ = extract(SphereSDF(radius=0.5))
    mesh_sdf = TriangleMeshSDF(v, f)
    p = random_points(1000, seed=4)
    q, tri = mesh_sdf.closest_points(p)
    assert q.shape == p.shape
    assert tri.shape == (len(p),)
    assert tri.min().item() >= 0 and tri.max().item() < len(f)
    # The returned points lie on the returned triangles.
    tri_v = v[f[tri].long()]
    normals = torch.nn.functional.normalize(
        torch.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0], dim=-1), dim=-1
    )
    assert ((q - tri_v[:, 0]) * normals).sum(-1).abs().max().item() < 1e-5
    assert torch.allclose((p - q).norm(dim=-1), mesh_sdf(p).abs(), atol=1e-5)


def test_mesh_sdf_round_trip_extraction():
    """Sampling a mesh into a grid and extracting again reproduces it."""
    (v, f), cell = extract(RotationOp(sdf=CuboidSDF(size=[1.0, 1.0, 1.0]), axis=[1, 1, 0], angle=30))
    mesh_sdf = TriangleMeshSDF(v, f)
    grid = isoext.UniformGrid([64, 64, 64])
    grid.set_values(mesh_sdf(grid.get_points()))
    v2, f2 = isoext.dual_contouring(grid)
    assert len(v2) > 0
    assert mesh_sdf(v2).abs().max().item() < 0.5 * cell


def test_mesh_sdf_batched_shape():
    (v, f), _ = extract(SphereSDF(radius=0.5), n=16)
    mesh_sdf = TriangleMeshSDF(v, f)
    p = random_points(24).reshape(2, 3, 4, 3)
    assert mesh_sdf(p).shape == (2, 3, 4)


def test_winding_number_closed_mesh():
    """Close to 1 inside and 0 outside, and the sign agrees with ray parity."""
    (v, f), _ = extract(SphereSDF(radius=0.5))
    sdf = TriangleMeshSDF(v, f)
    p = random_points(20000, seed=5)
    w = sdf.winding_number(p)
    r = p.norm(dim=-1)
    assert (w[r < 0.45] - 1).abs().max().item() < 0.05
    assert w[r > 0.55].abs().max().item() < 0.05
    parity = TriangleMeshSDF(v, f, sign="parity")
    assert torch.equal(torch.sign(sdf(p)), torch.sign(parity(p)))


def test_winding_number_matches_exact_sum():
    """The tree approximation stays close to the exact sum of solid angles."""
    (v, f), _ = extract(SphereSDF(radius=0.5), n=24)
    sdf = TriangleMeshSDF(v, f)
    p = random_points(300, seed=6)
    tri = v[f.long()].double()
    a, b, c = (tri[None, :, k] - p.double()[:, None] for k in range(3))
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    num = (a * torch.cross(b, c, dim=-1)).sum(-1)
    den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
    exact = (2 * torch.atan2(num, den)).sum(-1) / (4 * torch.pi)
    assert (sdf.winding_number(p) - exact.float()).abs().max().item() < 0.05


def test_winding_sign_survives_holes():
    """A mesh with a hole keeps the right sign away from the hole, where parity fails."""
    (v, f), cell = extract(SphereSDF(radius=0.5))
    # Cut a hole: drop the triangles around the north pole.
    centers = v[f.long()].mean(dim=1)
    keep = ~((centers[:, 2] > 0.35) & (centers[:, :2].norm(dim=-1) < 0.25))
    assert keep.sum() < len(f)
    holed = TriangleMeshSDF(v, f[keep])
    p = random_points(20000, seed=7)
    far_from_hole = (p - torch.tensor([0.0, 0.0, 0.5], device="cuda")).norm(dim=-1) > 0.35
    p = p[far_from_hole]
    r = p.norm(dim=-1)
    d = holed(p)
    assert (d[r < 0.45] < 0).all()
    assert (d[r > 0.55] > 0).all()


def test_winding_sign_is_orientation_independent():
    (v, f), _ = extract(SphereSDF(radius=0.5), n=24)
    flipped = TriangleMeshSDF(v, f[:, [0, 2, 1]])
    p = random_points(5000, seed=8)
    r = p.norm(dim=-1)
    d = flipped(p)
    assert (d[r < 0.45] < 0).all()
    assert (d[r > 0.55] > 0).all()
    assert (flipped.winding_number(p)[r < 0.45] + 1).abs().max().item() < 0.05
