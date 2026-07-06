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
import viser


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
            raise ImportError("Color names require matplotlib; pass an RGB tuple instead.") from e
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
    side: str = "front",
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
        side: Which triangle sides to render: "front", "back" or "double".
            Use "double" for open surfaces, which disappear from behind
            with the default backface culling.

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
        side=side,
    )


def add_grid(
    server,
    grid,
    *,
    level: float = 0.0,
    name: str = "/grid",
    point_size: float | None = None,
    line_width: float = 2.0,
):
    """Draw a grid's cell edges and its corner values as colored dots.

    Corners with a value below the level are drawn red (inside the
    surface), the rest blue. For sparse grids only the active cells are
    drawn. Meant for small demonstration grids, like the single-cell
    examples on the marching cubes variants page.

    Args:
        server: A viser.ViserServer instance.
        grid: A UniformGrid or SparseGrid whose values are set.
        level: The iso-value that separates inside from outside.
        name: Scene tree name prefix for the lines and dots.
        point_size: Dot diameter in world units. Defaults to a fraction
            of the cell edge length.
        line_width: Width of the cell edges in pixels.
    """
    import numpy as np

    points = grid.get_points()
    if points[..., 0].numel() > 32**3:
        raise ValueError("add_grid is meant for small demonstration grids")
    if points.ndim == 4:  # uniform: (nx, ny, nz, 3)
        p = points
        segments = torch.cat(
            [
                torch.stack([p[:-1], p[1:]], dim=-2).reshape(-1, 2, 3),
                torch.stack([p[:, :-1], p[:, 1:]], dim=-2).reshape(-1, 2, 3),
                torch.stack([p[:, :, :-1], p[:, :, 1:]], dim=-2).reshape(-1, 2, 3),
            ]
        )
        edge = (p[1, 0, 0] - p[0, 0, 0]).norm() if p.shape[0] > 1 else 1.0
    elif points.ndim == 3 and points.shape[1:] == (8, 3):  # sparse cells
        if len(points) == 0:
            return
        # Corner index bits within a cell are (x << 2) | (y << 1) | z.
        cell_edges = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        segments = points[:, cell_edges].reshape(-1, 2, 3)
        edge = (points[0, 1] - points[0, 0]).norm()
    else:
        raise ValueError("add_grid expects a UniformGrid or SparseGrid")

    segments = segments.cpu().numpy()
    server.scene.add_line_segments(
        f"{name}/edges",
        segments,
        colors=np.full_like(segments, 0.55),
        line_width=line_width,
    )

    if point_size is None:
        point_size = 0.1 * float(edge)
    inside = (grid.get_values() < level).reshape(-1).cpu().numpy()
    colors = np.where(inside[:, None], (0.85, 0.25, 0.2), (0.25, 0.45, 0.85)).astype(np.float32)
    server.scene.add_point_cloud(
        f"{name}/corners",
        points.reshape(-1, 3).cpu().numpy(),
        colors=colors,
        point_size=point_size,
    )


def show(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    port: int = 8080,
    grid=None,
    grid_level: float = 0.0,
    **mesh_kwargs,
):
    """Open an interactive viewer serving the given mesh.

    The server keeps running until it is stopped or the process exits; the
    printed URL can be opened in any browser.

    Args:
        vertices: (N, 3) tensor of vertex positions.
        faces: (M, 3) tensor of triangle indices.
        port: Port to serve on (the next free port is used if taken).
        grid: Optional grid to overlay with add_grid.
        grid_level: Iso-value for the grid overlay's corner colors.
        **mesh_kwargs: Forwarded to add_mesh.

    Returns:
        The running viser.ViserServer; call .stop() to shut it down.
    """
    server = viser.ViserServer(port=port)
    add_mesh(server, vertices, faces, **mesh_kwargs)
    if grid is not None:
        add_grid(server, grid, level=grid_level)
    return server


# Hidden server reused by serialize_scene. Stopping a server prints from its
# background thread, so keeping one alive is both quieter and faster than
# creating a fresh one per call.
_scene_recorder = None


def _get_scene_recorder():
    global _scene_recorder
    if _scene_recorder is None:
        with _suppress_output():
            _scene_recorder = viser.ViserServer(verbose=False)
    else:
        _scene_recorder.scene.reset()
    return _scene_recorder


def serialize_scene(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    grid=None,
    grid_level: float = 0.0,
    **mesh_kwargs,
) -> bytes:
    """Serialize a scene containing the given mesh to .viser bytes.

    The bytes can be written to a ``.viser`` file and played back offline by
    viser's static client, e.g. embedded in a web page. See save_scene and
    embed for convenience wrappers.
    """
    server = _get_scene_recorder()
    add_mesh(server, vertices, faces, **mesh_kwargs)
    if grid is not None:
        add_grid(server, grid, level=grid_level)
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
        **mesh_kwargs: Forwarded to add_mesh; pass grid= (and optionally
            grid_level=) to overlay the grid's edges and corner signs.

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
