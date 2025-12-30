"""Visualization utilities for isoext documentation."""

import numpy as np
import pyvista as pv

# Use static backend for documentation (creates inline images)
# This works in static HTML without a live kernel
pv.set_jupyter_backend("static")
pv.global_theme.window_size = [600, 400]
pv.global_theme.anti_aliasing = "ssaa"


def show_mesh(vertices, faces, **kwargs):
    """Display a mesh from isoext output.

    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) or (M, 4) tensor of face indices
        **kwargs: Additional arguments passed to pv.Plotter.add_mesh

    Returns:
        PyVista Plotter object
    """
    # Convert tensors to numpy
    verts = vertices.cpu().numpy()
    face_indices = faces.cpu().numpy()

    # Build PyVista-compatible face array
    # PyVista expects [n_verts, v0, v1, v2, ...] format
    n_verts_per_face = face_indices.shape[1]
    n_faces = face_indices.shape[0]
    
    # Vectorized construction of face array
    prefix = np.full((n_faces, 1), n_verts_per_face, dtype=np.int32)
    pv_faces = np.hstack([prefix, face_indices]).ravel()

    mesh = pv.PolyData(verts, pv_faces)

    # Default styling
    plot_kwargs = {
        "show_edges": False,
        "color": "lightblue",
        "smooth_shading": True,
    }
    plot_kwargs.update(kwargs)

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh, **plot_kwargs)
    pl.background_color = "white"
    pl.camera_position = "iso"
    pl.enable_anti_aliasing("ssaa")
    
    return pl.show(jupyter_backend="static")
