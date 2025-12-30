"""Shared test utilities and fixtures."""

import pytest

import isoext
from isoext.sdf import SphereSDF, TorusSDF


@pytest.fixture
def sphere():
    """A standard sphere SDF with radius 0.5."""
    return SphereSDF(radius=0.5)


@pytest.fixture
def torus():
    """A standard torus SDF with R=0.5, r=0.2."""
    return TorusSDF(R=0.5, r=0.2)


@pytest.fixture
def sphere_grid(sphere):
    """A 32x32x32 UniformGrid populated with sphere SDF values."""
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = sphere(grid.get_points())
    grid.set_values(sdf_values)
    return grid


@pytest.fixture
def torus_grid(torus):
    """A 32x32x32 UniformGrid populated with torus SDF values."""
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sdf_values = torus(grid.get_points())
    grid.set_values(sdf_values)
    return grid


def populate_sparse_grid(grid, sdf, shape, level=0.0):
    """Find surface-crossing cells and populate a SparseGrid with SDF values.

    Args:
        grid: A SparseGrid instance.
        sdf: An SDF function that takes points and returns SDF values.
        shape: The grid shape as (x, y, z).
        level: The iso-level to find surface-crossing cells for.
    """
    chunk_size = shape[0] * shape[1] * shape[2]
    chunks = grid.get_potential_cell_indices(chunk_size)

    for chunk in chunks:
        points = grid.get_points_by_cell_indices(chunk)
        sdf_values = sdf(points)
        filtered = grid.filter_cell_indices(chunk, sdf_values, level=level)
        if len(filtered) > 0:
            grid.add_cells(filtered)

    if grid.get_num_cells() > 0:
        points = grid.get_points()
        sdf_values = sdf(points)
        grid.set_values(sdf_values)
