"""Interactive mesh viewing and scene export built on viser.

Typical use::

    import isoext
    from isoext import viewer

    v, f = isoext.marching_cubes(grid)
    server = viewer.show(v, f)   # open an interactive viewer in the browser
    viewer.embed(v, f)           # inline scene for (statically hosted) notebooks
"""

import contextlib
import hashlib
import io
import shutil
from pathlib import Path

import torch


def _import_viser():
    try:
        import viser
    except ImportError as e:
        raise ImportError(
            "The isoext viewer requires viser. Install it with: pip install viser"
        ) from e
    return viser


def _suppress_output():
    """Hide chatter printed by viser during server startup.

    Inside notebooks, viser's banner is emitted as rich Jupyter display
    output rather than through stdout, so IPython's capture is needed when
    available.
    """
    try:
        from IPython.utils.capture import capture_output

        return capture_output()
    except ImportError:
        return contextlib.redirect_stdout(io.StringIO())


def _to_rgb(color):
    """Convert color names like "coral" to the RGB tuples viser expects."""
    if isinstance(color, str):
        try:
            from matplotlib.colors import to_rgb
        except ImportError as e:
            raise ImportError(
                "Color names require matplotlib; pass an RGB tuple instead."
            ) from e
        return to_rgb(color)
    return color


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
        color: RGB tuple with components in [0, 1], or a color name.
        flat_shading: Shade each triangle with a constant normal.
        wireframe: Render the mesh as a wireframe.

    Returns:
        The viser mesh handle.
    """
    return server.scene.add_mesh_simple(
        name,
        vertices.detach().cpu().numpy(),
        faces.detach().cpu().numpy(),
        color=_to_rgb(color),
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
        with _suppress_output():
            _scene_recorder = viser.ViserServer(verbose=False)
    else:
        _scene_recorder.scene.reset()
    return _scene_recorder


def serialize_scene(vertices: torch.Tensor, faces: torch.Tensor, **mesh_kwargs) -> bytes:
    """Serialize a scene containing the given mesh to .viser bytes.

    The bytes can be written to a ``.viser`` file and played back offline by
    viser's static client, e.g. embedded in a web page. See save_scene and
    embed for convenience wrappers.
    """
    server = _get_scene_recorder()
    add_mesh(server, vertices, faces, **mesh_kwargs)
    return server.get_scene_serializer().serialize()


def save_scene(path, vertices: torch.Tensor, faces: torch.Tensor, **mesh_kwargs) -> None:
    """Serialize a scene with the given mesh and write it to a .viser file."""
    Path(path).write_bytes(serialize_scene(vertices, faces, **mesh_kwargs))


def copy_client(directory) -> Path:
    """Copy viser's single-file static web client into a directory.

    The client is a self-contained index.html that plays back .viser scene
    files passed via its ``?playbackPath=`` URL parameter. The copy is
    refreshed whenever the installed viser ships a different build.

    Returns:
        The path of the copied index.html.
    """
    viser = _import_viser()
    src = Path(viser.__file__).parent / "client" / "build" / "index.html"
    dst = Path(directory) / "index.html"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    return dst


def embed(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    root="_static",
    height: int = 420,
    **mesh_kwargs,
):
    """Display a mesh as a self-contained interactive scene in a notebook.

    Unlike show, which starts a live server, this writes static assets --
    the viser client to root/viser/ and the serialized scene to
    root/scenes/ -- and returns an IFrame referencing them with relative
    URLs. Rendered notebooks therefore stay interactive when hosted
    statically, for example on documentation pages.

    Args:
        vertices: (N, 3) tensor of vertex positions.
        faces: (M, 3) tensor of triangle indices.
        root: Directory for the static assets, relative to the notebook.
        height: Height of the embedded viewer in pixels.
        **mesh_kwargs: Forwarded to add_mesh.

    Returns:
        An IPython IFrame displaying the scene.
    """
    try:
        from IPython.display import IFrame
    except ImportError as e:
        raise ImportError("embed is meant for notebooks and requires IPython.") from e

    root = Path(root)
    copy_client(root / "viser")

    data = serialize_scene(vertices, faces, **mesh_kwargs)
    # Content-addressed file names keep re-runs idempotent and let unchanged
    # cells keep referencing the same scene file.
    scenes_dir = root / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    scene_name = hashlib.sha1(data).hexdigest()[:16] + ".viser"
    (scenes_dir / scene_name).write_bytes(data)

    # Both URLs are relative: the iframe src is resolved against the page and
    # playbackPath against the client's own URL.
    src = f"{root.as_posix()}/viser/index.html?playbackPath=../scenes/{scene_name}"
    return IFrame(
        src,
        width="100%",
        height=height,
        extras=['style="border: 1px solid #8888; border-radius: 4px;"'],
    )
