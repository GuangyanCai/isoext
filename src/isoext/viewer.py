"""Interactive mesh viewing and scene export built on viser.

Requires the optional viser dependency::

    pip install isoext[viewer]

Typical use::

    import isoext
    from isoext import viewer

    v, f = isoext.marching_cubes(grid)
    server = viewer.show(v, f)   # open an interactive viewer in the browser
"""

import contextlib
import io
from pathlib import Path

import torch


def _import_viser():
    try:
        import viser
    except ImportError as e:
        raise ImportError(
            "The isoext viewer requires viser. Install it with: pip install isoext[viewer]"
        ) from e
    return viser


def add_mesh(
    server,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    name: str = "/mesh",
    color=(0.71, 0.8, 1.0),
    flat_shading: bool = False,
    wireframe: bool = False,
):
    """Add a mesh from isoext output tensors to a viser scene.

    Args:
        server: A viser.ViserServer instance.
        vertices: (N, 3) tensor of vertex positions.
        faces: (M, 3) tensor of triangle indices.
        name: Scene tree name of the mesh.
        color: RGB color with components in [0, 1].
        flat_shading: Shade each triangle with a constant normal.
        wireframe: Render the mesh as a wireframe.

    Returns:
        The viser mesh handle.
    """
    return server.scene.add_mesh_simple(
        name,
        vertices.detach().cpu().numpy(),
        faces.detach().cpu().numpy(),
        color=color,
        flat_shading=flat_shading,
        wireframe=wireframe,
    )


def show(vertices: torch.Tensor, faces: torch.Tensor, *, port: int = 8080, **mesh_kwargs):
    """Open an interactive viewer serving the given mesh.

    The server keeps running until it is stopped or the process exits; the
    printed URL can be opened in any browser.

    Args:
        vertices: (N, 3) tensor of vertex positions.
        faces: (M, 3) tensor of triangle indices.
        port: Port to serve on (the next free port is used if taken).
        **mesh_kwargs: Forwarded to add_mesh.

    Returns:
        The running viser.ViserServer; call .stop() to shut it down.
    """
    viser = _import_viser()
    server = viser.ViserServer(port=port)
    add_mesh(server, vertices, faces, **mesh_kwargs)
    return server


# Hidden server reused by serialize_scene. Stopping a server prints from its
# background thread, so keeping one alive is both quieter and faster than
# creating a fresh one per call.
_scene_recorder = None


def _get_scene_recorder():
    global _scene_recorder
    viser = _import_viser()
    if _scene_recorder is None:
        # The server prints a startup banner even with verbose=False; swallow
        # it since nothing ever connects to this instance.
        with contextlib.redirect_stdout(io.StringIO()):
            _scene_recorder = viser.ViserServer(verbose=False)
    else:
        _scene_recorder.scene.reset()
    return _scene_recorder


def serialize_scene(vertices: torch.Tensor, faces: torch.Tensor, **mesh_kwargs) -> bytes:
    """Serialize a scene containing the given mesh to .viser bytes.

    The bytes can be written to a ``.viser`` file and played back offline by
    viser's static client, e.g. embedded in a web page. See save_scene for a
    convenience wrapper.
    """
    server = _get_scene_recorder()
    add_mesh(server, vertices, faces, **mesh_kwargs)
    return server.get_scene_serializer().serialize()


def save_scene(path, vertices: torch.Tensor, faces: torch.Tensor, **mesh_kwargs) -> None:
    """Serialize a scene with the given mesh and write it to a .viser file."""
    Path(path).write_bytes(serialize_scene(vertices, faces, **mesh_kwargs))
