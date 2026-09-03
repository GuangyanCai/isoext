"""Tests for the test-mesh loader and its PLY/OBJ parsers."""

import urllib.error

import numpy as np
import pytest
import torch

import isoext
from isoext.assets import _read_obj, _read_ply, load_mesh


def test_read_ply_ascii_with_extra_properties():
    ply = b"""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property float confidence
element face 2
property list uchar int vertex_indices
end_header
0 0 0 0.5
1 0 0 0.5
1 1 0 0.5
0 1 0 0.5
3 0 1 2
3 0 2 3
"""
    v, f = _read_ply(ply)
    assert v.shape == (4, 3) and v.dtype == np.float32
    assert f.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_read_ply_ascii_quads_are_triangulated():
    ply = b"""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
1 1 0
0 1 0
4 0 1 2 3
"""
    _, f = _read_ply(ply)
    assert f.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_read_ply_binary_big_endian_with_face_scalar():
    """The Armadillo layout: big-endian, a uchar before each face's index list."""
    header = b"""ply
format binary_big_endian 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property uchar intensity
property list uchar int vertex_indices
end_header
"""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=">f4").tobytes()
    face = (
        np.array([7], dtype="u1").tobytes()
        + np.array([3], dtype="u1").tobytes()
        + np.array([0, 1, 2], dtype=">i4").tobytes()
    )
    v, f = _read_ply(header + verts + face)
    assert v.tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    assert f.tolist() == [[0, 1, 2]]


def test_read_obj_quads_with_texture_indices():
    obj = b"""v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
f 1/1 2/1 3/1 4/1
"""
    v, f = _read_obj(obj)
    assert v.shape == (4, 3)
    assert f.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_load_mesh_unknown_name():
    with pytest.raises(KeyError):
        load_mesh("teapot")


def test_load_spot():
    """Spot is small and public domain; skipped without network access."""
    try:
        v, f = load_mesh("spot")
    except urllib.error.URLError:
        pytest.skip("no network access")
    assert v.shape == (2930, 3) and v.dtype == torch.float32
    assert f.shape[1] == 3 and f.dtype == torch.int32
    # Centered, longest side 1.6, and z-up (Spot stands taller than it is wide).
    lo, hi = v.min(0).values, v.max(0).values
    assert torch.allclose(lo + hi, torch.zeros(3, device=v.device), atol=1e-5)
    assert abs((hi - lo).max().item() - 1.6) < 1e-5
    # Closed: every edge has two triangles.
    edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]).sort(dim=-1).values
    _, counts = torch.unique(edges, dim=0, return_counts=True)
    assert (counts == 2).all()
    # Usable as an SDF: the mesh is recovered from its own distance field.
    sdf = isoext.sdf.TriangleMeshSDF(v, f)
    grid = isoext.UniformGrid([64, 64, 64])
    grid.set_values(sdf(grid.get_points()))
    v2, f2 = isoext.marching_cubes(grid)
    assert sdf(v2).abs().max().item() < 0.5 * (2 / 63)
