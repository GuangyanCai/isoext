"""Tests for dual contouring algorithm."""

import gc

import torch

import isoext
from isoext.sdf import (
    CuboidSDF,
    RotationOp,
    SphereSDF,
    get_sdf_normal,
    project_to_surface,
)

from conftest import populate_sparse_grid


def test_dual_contouring_simple(sphere_grid):
    """Test simplest dual contouring usage - no intersection provided."""
    v, f = isoext.dual_contouring(sphere_grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_vertex_accuracy(sphere_grid):
    """Dual vertices from the QEF solve must lie on the sphere surface.

    Guards the batched SVD path: a silently failed solve would leave
    vertices near cell corners instead (error on the order of a cell).
    """
    v, f = isoext.dual_contouring(sphere_grid, level=0.0)

    err = (v.norm(dim=-1) - 0.5).abs()
    cell_size = 2.0 / 31
    assert err.max().item() < cell_size / 4


def test_dual_contouring_sharp_features_with_sdf_normals():
    """Refined points and SDF normals must place vertices on sharp features.

    A rotated cube stresses vertex placement. Almost all vertices must sit
    on the surface; cells whose crease lies in a neighboring cell carry a
    bounded error, since one vertex per cell cannot span two face strips.
    """
    sdf = RotationOp(sdf=CuboidSDF(size=[1.0, 1.0, 1.0]), axis=[1, 1, 0], angle=30)
    grid = isoext.UniformGrid([48, 48, 48], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    grid.set_values(sdf(grid.get_points()))
    cell_size = 2.0 / 47

    its = isoext.get_intersection(grid)
    points = project_to_surface(sdf, its.get_points())
    its.set_points(points)
    its.set_normals(get_sdf_normal(sdf, points))
    v, f = isoext.dual_contouring(grid, intersection=its)

    err = sdf(v).abs()
    assert err.quantile(0.99).item() < 0.08 * cell_size
    assert err.max().item() < 0.25 * cell_size

    # Unclamped vertices may leave their cells to sit on the features.
    v, f = isoext.dual_contouring(grid, intersection=its, clamp=False)
    err = sdf(v).abs()
    assert err.max().item() < 0.05 * cell_size


def test_dual_contouring_with_intersection_auto_normals(sphere_grid):
    """Test dual contouring with intersection but normals computed automatically."""
    # Get intersection without computing normals
    its = isoext.get_intersection(sphere_grid, level=0.0, compute_normals=False)
    assert not its.has_normals()

    # Dual contouring should compute normals automatically
    v, f = isoext.dual_contouring(sphere_grid, level=0.0, intersection=its)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_with_custom_normals(sphere, sphere_grid):
    """Test dual contouring with user-provided normals."""
    # Get intersection points
    its = isoext.get_intersection(sphere_grid, level=0.0)
    points = its.get_points()

    # Compute custom normals using SDF gradient
    normals = get_sdf_normal(sphere, points)
    its.set_normals(normals)
    assert its.has_normals()

    # Run dual contouring with custom normals
    v, f = isoext.dual_contouring(sphere_grid, level=0.0, intersection=its)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_different_levels(sphere_grid):
    """Test dual contouring with different iso-levels."""
    for level in [-0.1, 0.0, 0.1]:
        v, f = isoext.dual_contouring(sphere_grid, level=level)
        if len(v) > 0:
            assert v.shape[1] == 3
            assert f.shape[1] == 3


def test_dual_contouring_empty_result():
    """Test dual contouring when no surface is found."""
    grid = isoext.UniformGrid([8, 8, 8], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    grid.set_values(torch.ones((8, 8, 8), device="cuda"))

    v, f = isoext.dual_contouring(grid, level=0.0)

    # No surface must yield empty tensors, not None.
    assert isinstance(v, torch.Tensor)
    assert isinstance(f, torch.Tensor)
    assert v.shape == (0, 3)
    assert f.shape == (0, 3)


def test_dual_contouring_parameters(sphere_grid):
    """Test dual contouring with different regularization parameters."""
    v, f = isoext.dual_contouring(sphere_grid, level=0.0, reg=0.01)

    assert v.shape[1] == 3
    assert f.shape[1] == 3


def test_dual_contouring_surface_crossing_domain_boundary():
    """Test dual contouring when the surface extends past the grid AABB.

    Edges on the max boundary faces have fewer than 4 neighboring cells;
    they must be skipped instead of producing out-of-range cell indices.
    """
    sphere = SphereSDF(radius=1.2)
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    grid.set_values(sphere(grid.get_points()))

    v, f = isoext.dual_contouring(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert f.max().item() < len(v)

    # Connectivity must only link dual vertices of adjacent cells, never
    # distant vertices across the domain.
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


def test_get_intersection(sphere_grid):
    """Test getting intersection points from grid."""
    its = isoext.get_intersection(sphere_grid, level=0.0)
    points = its.get_points()

    assert points.shape[1] == 3
    assert len(points) > 0
    assert not its.has_normals()  # Normals not computed by default


def test_get_intersection_with_normals(sphere_grid):
    """Test getting intersection with normals computed."""
    its = isoext.get_intersection(sphere_grid, level=0.0, compute_normals=True)
    points = its.get_points()
    normals = its.get_normals()

    assert points.shape[1] == 3
    assert normals.shape[1] == 3
    assert len(points) > 0
    assert its.has_normals()


def test_intersection_set_normals(sphere, sphere_grid):
    """Test setting normals on intersection object."""
    its = isoext.get_intersection(sphere_grid, level=0.0)
    assert not its.has_normals()

    points = its.get_points()
    if len(points) > 0:
        normals = get_sdf_normal(sphere, points)
        its.set_normals(normals)
        assert its.has_normals()

        # Verify normals were set
        retrieved_normals = its.get_normals()
        assert retrieved_normals.shape == normals.shape
        assert torch.allclose(retrieved_normals, normals, atol=1e-5)


def test_intersection_getters_do_not_alias_internal_storage(sphere, sphere_grid):
    """get_points/get_normals must return independent copies.

    Regression test: the first call used to transfer ownership of the
    internal buffer to the returned tensor, so set_normals silently
    overwrote tensors held by the caller, and dropping a returned tensor
    freed memory the Intersection still referenced (use-after-free).
    """
    its = isoext.get_intersection(sphere_grid, level=0.0, compute_normals=True)

    # The returned tensor must not alias the internal normals storage.
    n1 = its.get_normals()
    snapshot = n1.clone()
    its.set_normals(torch.full_like(n1, 0.5))
    assert torch.equal(n1, snapshot)

    # Dropping returned tensors must not free the intersection's storage.
    p1 = its.get_points()
    expected_points = p1.clone()
    del p1, n1
    gc.collect()
    # Churn device memory to encourage reuse of any wrongly-freed block.
    its2 = isoext.get_intersection(sphere_grid, level=0.1, compute_normals=True)
    assert its2.get_points().shape[1] == 3
    assert torch.equal(its.get_points(), expected_points)

    # The intersection must still be fully usable for dual contouring.
    normals = get_sdf_normal(sphere, expected_points)
    its.set_normals(normals)
    v, f = isoext.dual_contouring(sphere_grid, level=0.0, intersection=its)
    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0


def test_dual_contouring_non_uniform_resolution(sphere):
    """Test dual contouring with non-uniform resolution."""
    grid = isoext.UniformGrid([16, 32, 48], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.dual_contouring(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_non_uniform_resolution_extreme(sphere):
    """Test dual contouring with extreme non-uniform resolution ratios."""
    grid = isoext.UniformGrid([8, 64, 16], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    v, f = isoext.dual_contouring(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_get_intersection_non_uniform_resolution(sphere):
    """Test getting intersection points from non-uniform resolution grid."""
    grid = isoext.UniformGrid([16, 32, 48], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)

    its = isoext.get_intersection(grid, level=0.0, compute_normals=True)
    points = its.get_points()
    normals = its.get_normals()

    assert points.shape[1] == 3
    assert normals.shape[1] == 3
    assert its.has_normals()
    assert len(points) > 0


def test_dual_contouring_sparse_grid(sphere):
    """Test dual contouring with SparseGrid."""
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    v, f = isoext.dual_contouring(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_sparse_grid_with_custom_normals(sphere):
    """Test dual contouring with SparseGrid and custom normals."""
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    # Get intersection and compute custom normals
    its = isoext.get_intersection(grid, level=0.0)
    its_points = its.get_points()
    normals = get_sdf_normal(sphere, its_points)
    its.set_normals(normals)
    assert its.has_normals()

    v, f = isoext.dual_contouring(grid, level=0.0, intersection=its)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert len(f) > 0


def test_dual_contouring_sparse_grid_surface_crossing_domain_boundary():
    """Test sparse-grid dual contouring when the surface extends past the AABB.

    Boundary edges are marked with -1 (no neighbor cell); the sparse-to-uniform
    index remap must preserve the markers instead of dereferencing them.
    """
    shape = [32, 32, 32]
    sphere = SphereSDF(radius=1.2)
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    v, f = isoext.dual_contouring(grid, level=0.0)

    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert len(v) > 0
    assert f.max().item() < len(v)

    # Dual vertices are clipped to their cell AABB, so every vertex must lie
    # inside the domain and within a cell diagonal of the sphere surface.
    cell_size = 2.0 / 31
    assert v.abs().max().item() <= 1.0 + 1e-5
    err = (v.norm(dim=-1) - 1.2).abs()
    assert err.max().item() < 2 * cell_size

    # Connectivity must only link dual vertices of adjacent cells.
    tri = v[f.long()]
    edge_len = torch.cat(
        [
            (tri[:, 0] - tri[:, 1]).norm(dim=-1),
            (tri[:, 1] - tri[:, 2]).norm(dim=-1),
            (tri[:, 2] - tri[:, 0]).norm(dim=-1),
        ]
    )
    assert edge_len.max().item() < 4 * cell_size


def test_get_intersection_sparse_grid(sphere):
    """Test getting intersection points from SparseGrid."""
    shape = [32, 32, 32]
    grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    populate_sparse_grid(grid, sphere, shape, level=0.0)

    its = isoext.get_intersection(grid, level=0.0, compute_normals=True)
    its_points = its.get_points()
    its_normals = its.get_normals()

    assert its_points.shape[1] == 3
    assert its_normals.shape[1] == 3
    assert its.has_normals()
    assert len(its_points) > 0


def test_dual_contouring_sparse_grid_different_levels(sphere):
    """Test dual contouring with SparseGrid at different iso-levels."""
    shape = [32, 32, 32]

    for level in [-0.1, 0.0, 0.1]:
        grid = isoext.SparseGrid(shape, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
        populate_sparse_grid(grid, sphere, shape, level=level)

        if grid.get_num_cells() > 0:
            v, f = isoext.dual_contouring(grid, level=level)

            assert v.shape[1] == 3
            assert f.shape[1] == 3
