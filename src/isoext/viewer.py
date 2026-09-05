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
import itertools
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


# Distinct default scene names for anonymous annotations; viser replaces
# nodes that share a name.
_anon_names = itertools.count()


def add_points(
    server,
    points: torch.Tensor,
    *,
    color=(0.93, 0.79, 0.24),
    point_size: float = 0.06,
    name: str | None = None,
):
    """Draw raw points, for annotating a scene on top of the mesh and grid.

    Args:
        server: A viser.ViserServer instance.
        points: (N, 3) tensor of positions.
        color: RGB tuple with components in [0, 1], or a color name.
        point_size: Dot diameter in world units.
        name: Scene tree name of the point cloud. Defaults to a unique
            name, so repeated calls add to the scene instead of replacing
            the previous points.
    """
    import numpy as np

    if name is None:
        name = f"/points/{next(_anon_names)}"
    pts = points.detach().reshape(-1, 3).cpu().numpy()
    colors = np.tile(np.asarray(_to_rgb(color), dtype=np.float32), (len(pts), 1))
    return server.scene.add_point_cloud(name, pts, colors=colors, point_size=point_size)


def add_lines(
    server,
    segments: torch.Tensor,
    *,
    color=(0.55, 0.55, 0.55),
    line_width: float = 2.0,
    name: str | None = None,
):
    """Draw raw line segments, for annotating a scene.

    Args:
        server: A viser.ViserServer instance.
        segments: (N, 2, 3) tensor: N segments with start and end points.
        color: RGB tuple with components in [0, 1], or a color name.
        line_width: Width of the segments in pixels.
        name: Scene tree name of the segments. Defaults to a unique name,
            so repeated calls add to the scene instead of replacing the
            previous segments.
    """
    import numpy as np

    if name is None:
        name = f"/lines/{next(_anon_names)}"
    segs = segments.detach().reshape(-1, 2, 3).cpu().numpy()
    colors = np.broadcast_to(np.asarray(_to_rgb(color), dtype=np.float32), segs.shape)
    return server.scene.add_line_segments(name, segs, colors=colors, line_width=line_width)


def add_label(server, text: str, position, *, name: str | None = None):
    """Draw a text label, e.g. to caption the meshes of a composed scene.

    Args:
        server: A viser.ViserServer instance.
        text: The label text.
        position: The label position as a (3,) tensor, list or tuple.
        name: Scene tree name. Defaults to a unique name.
    """
    if name is None:
        name = f"/labels/{next(_anon_names)}"
    if isinstance(position, torch.Tensor):
        position = position.detach().reshape(3).cpu().tolist()
    return server.scene.add_label(name, text=text, position=tuple(position))


def _orthonormal_frame(directions):
    """Unit direction plus two unit vectors spanning its perpendicular plane."""
    n = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    helper = torch.zeros_like(n)
    helper[..., 0] = 1.0
    helper[n[..., 0].abs() > 0.9] = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)
    u = torch.cross(n, helper, dim=-1)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return n, u, torch.cross(n, u, dim=-1)


def add_arrows(
    server,
    origins: torch.Tensor,
    directions: torch.Tensor,
    *,
    color=(0.45, 0.45, 0.45),
    line_width: float = 2.0,
    name: str | None = None,
):
    """Draw arrows as line shafts with cone heads, e.g. for normals.

    Args:
        server: A viser.ViserServer instance.
        origins: (N, 3) tensor of arrow start points.
        directions: (N, 3) tensor of arrow vectors; length sets the size.
        color: RGB tuple with components in [0, 1], or a color name.
        line_width: Width of the shafts in pixels.
        name: Scene tree name prefix. Defaults to a unique name.
    """
    import numpy as np

    if name is None:
        name = f"/arrows/{next(_anon_names)}"
    origins = origins.detach().reshape(-1, 3)
    directions = directions.detach().reshape(-1, 3)
    tips = origins + directions
    n, u, w = _orthonormal_frame(directions)
    head = 0.25 * directions.norm(dim=-1, keepdim=True)
    base = tips - head * n
    radius = 0.4 * head

    count = len(origins)
    k = 12
    angles = torch.arange(k, device=origins.device) * (2.0 * torch.pi / k)
    ring = base[:, None] + radius[:, None] * (
        torch.cos(angles)[None, :, None] * u[:, None] + torch.sin(angles)[None, :, None] * w[:, None]
    )
    verts = torch.cat([ring.reshape(-1, 3), tips, base])
    j = torch.arange(k, device=origins.device)
    jn = (j + 1) % k
    i = torch.arange(count, device=origins.device)[:, None]
    apex = (count * k + i).expand(-1, k)
    bottom = (count * k + count + i).expand(-1, k)
    faces = torch.cat(
        [
            torch.stack([i * k + j, i * k + jn, apex], dim=-1).reshape(-1, 3),
            torch.stack([i * k + jn, i * k + j, bottom], dim=-1).reshape(-1, 3),
        ]
    )
    server.scene.add_mesh_simple(
        f"{name}/heads",
        verts.cpu().numpy(),
        faces.cpu().numpy(),
        color=_to_rgb(color),
    )
    segments = torch.stack([origins, base], dim=1).cpu().numpy()
    server.scene.add_line_segments(
        f"{name}/shafts",
        segments,
        colors=np.broadcast_to(np.asarray(_to_rgb(color), dtype=np.float32), segments.shape),
        line_width=line_width,
    )


def add_planes(
    server,
    centers: torch.Tensor,
    normals: torch.Tensor,
    *,
    size: float = 0.5,
    color=(0.6, 0.7, 0.9),
    opacity: float = 0.4,
    name: str | None = None,
):
    """Draw translucent square patches perpendicular to the given normals,
    e.g. tangent planes.

    Args:
        server: A viser.ViserServer instance.
        centers: (N, 3) tensor of patch centers.
        normals: (N, 3) tensor of patch normals.
        size: Half of the patch side length, in world units.
        color: RGB tuple with components in [0, 1], or a color name.
        opacity: Patch opacity in [0, 1].
        name: Scene tree name. Defaults to a unique name.
    """
    if name is None:
        name = f"/planes/{next(_anon_names)}"
    centers = centers.detach().reshape(-1, 3)
    _, u, w = _orthonormal_frame(normals.detach().reshape(-1, 3))
    corners = torch.stack(
        [
            centers - size * u - size * w,
            centers + size * u - size * w,
            centers + size * u + size * w,
            centers - size * u + size * w,
        ],
        dim=1,
    )
    i = torch.arange(len(centers), device=centers.device)[:, None] * 4
    tri = torch.tensor([[0, 1, 2], [0, 2, 3]], device=centers.device)
    faces = (i[:, None] + tri[None]).reshape(-1, 3)
    server.scene.add_mesh_simple(
        name,
        corners.reshape(-1, 3).cpu().numpy(),
        faces.cpu().numpy(),
        color=_to_rgb(color),
        opacity=opacity,
        side="double",
    )


def add_spheres(
    server,
    centers: torch.Tensor,
    radii: torch.Tensor,
    *,
    color=(0.6, 0.7, 0.9),
    opacity: float = 0.3,
    name: str | None = None,
):
    """Draw translucent spheres, e.g. the distance spheres of SDF samples.

    Args:
        server: A viser.ViserServer instance.
        centers: (N, 3) tensor of sphere centers.
        radii: (N,) tensor of radii.
        color: RGB tuple with components in [0, 1], or a color name.
        opacity: Sphere opacity in [0, 1].
        name: Scene tree name. Defaults to a unique name.
    """
    if name is None:
        name = f"/spheres/{next(_anon_names)}"
    centers = centers.detach().reshape(-1, 3)
    radii = radii.detach().reshape(-1).to(centers)
    # One unit UV sphere, instanced per center.
    rings, segments = 12, 24
    theta = torch.linspace(0, torch.pi, rings + 1, device=centers.device)
    phi = torch.linspace(0, 2 * torch.pi, segments + 1, device=centers.device)[:-1]
    unit = torch.stack(
        [
            torch.sin(theta)[:, None] * torch.cos(phi)[None],
            torch.sin(theta)[:, None] * torch.sin(phi)[None],
            torch.cos(theta)[:, None].expand(-1, segments),
        ],
        dim=-1,
    ).reshape(-1, 3)
    r = torch.arange(rings, device=centers.device)[:, None] * segments
    c = torch.arange(segments, device=centers.device)[None]
    c1 = (c + 1) % segments
    quads = torch.stack([r + c, r + segments + c, r + segments + c1, r + c1], dim=-1).reshape(-1, 4)
    unit_faces = torch.cat([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])

    verts = centers[:, None] + radii[:, None, None] * unit[None]
    offsets = torch.arange(len(centers), device=centers.device)[:, None, None] * len(unit)
    faces = (unit_faces[None] + offsets).reshape(-1, 3)
    server.scene.add_mesh_simple(
        name,
        verts.reshape(-1, 3).cpu().numpy(),
        faces.cpu().numpy(),
        color=_to_rgb(color),
        opacity=opacity,
    )


def show(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    port: int = 8080,
    grid=None,
    grid_level: float = 0.0,
    draw=None,
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
        draw: Callable that receives the server to add extra elements,
            e.g. with add_points and add_lines.
        **mesh_kwargs: Forwarded to add_mesh.

    Returns:
        The running viser.ViserServer; call .stop() to shut it down.
    """
    server = viser.ViserServer(port=port)
    add_mesh(server, vertices, faces, **mesh_kwargs)
    if grid is not None:
        add_grid(server, grid, level=grid_level)
    if draw is not None:
        draw(server)
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
    vertices: torch.Tensor | None = None,
    faces: torch.Tensor | None = None,
    *,
    grid=None,
    grid_level: float = 0.0,
    draw=None,
    **mesh_kwargs,
) -> bytes:
    """Serialize a scene containing the given mesh to .viser bytes.

    The bytes can be written to a ``.viser`` file and played back offline by
    viser's static client, e.g. embedded in a web page. See save_scene and
    embed for convenience wrappers. The mesh is optional: pass None to
    build a scene of only a grid overlay and drawn annotations.
    """
    server = _get_scene_recorder()
    if vertices is not None:
        add_mesh(server, vertices, faces, **mesh_kwargs)
    if grid is not None:
        add_grid(server, grid, level=grid_level)
    if draw is not None:
        draw(server)
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


def _camera_params(vertices, grid, frame=None) -> str:
    """Initial camera URL parameters framing the mesh and grid.

    Without these the static client starts at a fixed distance, which
    leaves small scenes occupying a fraction of the viewport.
    """
    pts = []
    if frame is not None:
        pts.append(frame.detach().reshape(-1, 3))
    else:
        if vertices is not None and len(vertices) > 0:
            pts.append(vertices.detach().reshape(-1, 3))
        if grid is not None:
            pts.append(grid.get_points().detach().reshape(-1, 3))
    if not pts:
        return ""
    stacked = torch.cat(pts)
    low, high = stacked.amin(dim=0), stacked.amax(dim=0)
    center = ((low + high) / 2).cpu()
    radius = max(float(((high - low) / 2).norm()), 1e-3)
    direction = torch.tensor([1.0, 1.0, 0.7])
    position = center + 1.8 * radius * direction / direction.norm()
    look_at = ",".join(f"{v:.3f}" for v in center.tolist())
    pos = ",".join(f"{v:.3f}" for v in position.tolist())
    return f"&initialCameraPosition={pos}&initialCameraLookAt={look_at}&initialCameraUp=0,0,1"


def embed(
    vertices: torch.Tensor | None = None,
    faces: torch.Tensor | None = None,
    *,
    root="_static",
    height: int = 420,
    frame=None,
    **mesh_kwargs,
):
    """Display a mesh as a self-contained interactive scene in a notebook.

    Unlike show, which starts a live server, this writes static assets --
    the viser client to root/viser/ and the serialized scene to
    root/scenes/ -- and returns an IFrame referencing them with relative
    URLs. Rendered notebooks therefore stay interactive when hosted
    statically, for example on documentation pages.

    Args:
        vertices: (N, 3) tensor of vertex positions, or None for a scene
            without a mesh.
        faces: (M, 3) tensor of triangle indices.
        root: Directory for the static assets, relative to the notebook.
        height: Height of the embedded viewer in pixels.
        frame: Optional (N, 3) tensor of points that define the initial
            camera framing, for scenes composed with draw= whose extent
            the mesh alone does not describe.
        **mesh_kwargs: Forwarded to add_mesh; pass grid= (and optionally
            grid_level=) to overlay the grid's edges and corner signs, and
            draw= to add extra elements with add_points and add_lines.

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
    src += _camera_params(vertices, mesh_kwargs.get("grid"), frame)
    return IFrame(
        src,
        width="100%",
        height=height,
        extras=['style="border: 1px solid #8888; border-radius: 4px;"'],
    )
