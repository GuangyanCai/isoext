"""Visualization utilities for isoext documentation.

Meshes are rendered with viser through isoext.viewer: each show_mesh call
serializes the scene to a .viser file under _static/scenes/ and embeds
viser's static client in an iframe. Since both the client and the scene
files are plain static assets, the interactive viewer keeps working in the
statically hosted documentation.
"""

import hashlib
import shutil
from pathlib import Path

import viser
from IPython.display import IFrame
from matplotlib.colors import to_rgb

from isoext.viewer import serialize_scene

_STATIC_DIR = Path(__file__).parent / "_static"
_CLIENT_FILE = _STATIC_DIR / "viser" / "index.html"
_SCENES_DIR = _STATIC_DIR / "scenes"

_IFRAME_STYLE = "border: 1px solid #8888; border-radius: 4px;"


def ensure_client() -> None:
    """Copy viser's single-file static client into _static/viser/.

    The copy is refreshed whenever the installed viser version ships a
    different build. It is gitignored; sphinx bundles it into the built
    documentation together with the rest of _static.
    """
    src = Path(viser.__file__).parent / "client" / "build" / "index.html"
    if not _CLIENT_FILE.exists() or _CLIENT_FILE.stat().st_size != src.stat().st_size:
        _CLIENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, _CLIENT_FILE)


def show_mesh(vertices, faces, color="lightblue", smooth_shading=True, **kwargs):
    """Display a mesh from isoext output in an interactive viewer.

    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of face indices
        color: Color name or RGB tuple
        smooth_shading: Interpolate normals instead of flat triangle shading
        **kwargs: Additional arguments passed to isoext.viewer.add_mesh

    Returns:
        An HTML iframe embedding the viser client with the recorded scene.
    """
    ensure_client()

    data = serialize_scene(
        vertices, faces, color=to_rgb(color), flat_shading=not smooth_shading, **kwargs
    )

    # Content-addressed file names keep re-runs idempotent and let unchanged
    # cells keep referencing the same scene file.
    _SCENES_DIR.mkdir(parents=True, exist_ok=True)
    scene_name = hashlib.sha1(data).hexdigest()[:16] + ".viser"
    (_SCENES_DIR / scene_name).write_bytes(data)

    # Both paths are relative: the iframe src is resolved against the built
    # page (at the doc root) and playbackPath against the client's URL.
    src = f"_static/viser/index.html?playbackPath=../scenes/{scene_name}"
    return IFrame(src, width="100%", height=420, extras=[f'style="{_IFRAME_STYLE}"'])
