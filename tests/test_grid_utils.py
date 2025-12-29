"""Tests for grid utility functions."""

import os
import tempfile

import torch

import isoext
from isoext.sdf import SphereSDF
from isoext.utils import make_grid, write_obj


def test_make_grid_uniform():
    """Test make_grid with uniform resolution."""
    grid = make_grid([-1, -1, -1, 1, 1, 1], res=32, device="cuda")

    assert grid.shape == (32, 32, 32, 3)
    assert grid.device.type == "cuda"

    # Check bounds
    assert torch.allclose(grid[0, 0, 0], torch.tensor([-1.0, -1.0, -1.0], device="cuda"))
    assert torch.allclose(grid[-1, -1, -1], torch.tensor([1.0, 1.0, 1.0], device="cuda"), atol=1e-5)


def test_make_grid_non_uniform():
    """Test make_grid with non-uniform resolution."""
    grid = make_grid([-1, -1, -1, 1, 1, 1], res=[16, 32, 64], device="cuda")

    assert grid.shape == (16, 32, 64, 3)
    assert grid.device.type == "cuda"


def test_make_grid_cpu():
    """Test make_grid on CPU."""
    grid = make_grid([-1, -1, -1, 1, 1, 1], res=16, device="cpu")

    assert grid.shape == (16, 16, 16, 3)
    assert grid.device.type == "cpu"


def test_write_obj():
    """Test writing OBJ file."""
    v = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        device="cuda",
    )
    f = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], device="cuda")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_obj(tmp_path, v, f)

        # Verify file was created and has content
        assert os.path.exists(tmp_path)
        with open(tmp_path, "r") as f:
            content = f.read()
            assert "v " in content
            assert "f " in content
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_write_obj_empty():
    """Test writing empty OBJ file."""
    v = torch.empty((0, 3), device="cuda")
    f = torch.empty((0, 3), device="cuda", dtype=torch.long)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_obj(tmp_path, v, f)
        # Should not raise error
        assert os.path.exists(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_uniform_grid_basic():
    """Test UniformGrid basic functionality."""
    grid = isoext.UniformGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])

    assert grid.get_num_cells() > 0
    assert grid.get_num_points() > 0

    points = grid.get_points()
    # Points can be reshaped, check last dimension is 3
    assert points.shape[-1] == 3

    values = grid.get_values()
    assert len(values.shape) == 3


def test_uniform_grid_set_values():
    """Test setting values on UniformGrid."""
    grid = isoext.UniformGrid([16, 16, 16], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    sphere = SphereSDF(radius=0.5)

    points = grid.get_points()
    sdf_values = sphere(points)
    grid.set_values(sdf_values)

    retrieved_values = grid.get_values()
    assert retrieved_values.shape == sdf_values.shape
    assert torch.allclose(retrieved_values, sdf_values)


def test_uniform_grid_different_aabb():
    """Test UniformGrid with different bounding boxes."""
    grid = isoext.UniformGrid([16, 16, 16], aabb_min=[-2, -2, -2], aabb_max=[2, 2, 2])

    points = grid.get_points()
    # Check that points are within bounds
    assert torch.all(points >= -2)
    assert torch.all(points <= 2)


def test_sparse_grid_basic():
    """Test SparseGrid basic functionality."""
    grid = isoext.SparseGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])

    assert grid.get_num_cells() == 0  # Initially empty
    assert grid.get_num_points() == 0


def test_sparse_grid_add_cells():
    """Test adding cells to SparseGrid."""
    grid = isoext.SparseGrid([32, 32, 32], aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])

    # Add some cell indices
    cell_indices = torch.tensor([0, 100, 200], device="cuda", dtype=torch.int32)
    grid.add_cells(cell_indices)

    assert grid.get_num_cells() > 0

    cell_indices_retrieved = grid.get_cell_indices()
    assert len(cell_indices_retrieved) > 0
