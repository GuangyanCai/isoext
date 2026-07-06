"""Generate the interactive scenes embedded on the landing page.

Run manually after changing the shapes:

    pixi run --environment doc python _hero.py
"""

import time
from pathlib import Path

import torch

import isoext
from isoext.sdf import MandelbulbSDF
from isoext.utils import gaussian_smooth
from isoext.viewer import save_scene

SCENES = Path(__file__).parent / "_static" / "scenes"


def main() -> None:
    SCENES.mkdir(parents=True, exist_ok=True)

    # Hero: Mandelbulb from a raw PyTorch tensor field. Few fractal
    # iterations plus a light gaussian blur keep the surface detail above
    # the grid resolution, so the mesh stays smooth instead of noisy.
    grid = isoext.UniformGrid([192] * 3, aabb_min=[-1.2] * 3, aabb_max=[1.2] * 3)
    values = MandelbulbSDF(iterations=6)(grid.get_points())
    grid.set_values(gaussian_smooth(values, sigma=1.0))
    isoext.marching_cubes(grid)   # warm up: the first call pays the PTX JIT
    torch.cuda.synchronize()
    start = time.perf_counter()
    v, f = isoext.marching_cubes(grid)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    print(f"mandelbulb: {v.shape[0]:,} vertices, extracted in {elapsed_ms:.1f} ms")
    save_scene(SCENES / "hero.viser", v, f, color=(0.88, 0.45, 0.34))
    print(f"hero.viser: {(SCENES / 'hero.viser').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
