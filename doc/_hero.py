"""Generate the interactive scenes embedded on the landing page.

Run manually after changing the shapes:

    pixi run --environment doc python _hero.py
"""

import time
from pathlib import Path

import torch

import isoext
from isoext.utils import gaussian_smooth
from isoext.viewer import save_scene

SCENES = Path(__file__).parent / "_static" / "scenes"


def mandelbulb_field(points: torch.Tensor, power: float = 8.0, iterations: int = 10) -> torch.Tensor:
    """Distance estimator of the power-8 Mandelbulb, in plain PyTorch."""
    p = points.reshape(-1, 3)
    z = p.clone()
    dr = torch.ones(p.shape[0], device=p.device)
    r = z.norm(dim=-1)
    for _ in range(iterations):
        r = z.norm(dim=-1).clamp_min(1e-9)
        escaped = r > 2.0
        theta = torch.acos((z[:, 2] / r).clamp(-1.0, 1.0)) * power
        phi = torch.atan2(z[:, 1], z[:, 0]) * power
        zr = r**power
        z_next = (
            zr[:, None]
            * torch.stack(
                [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()],
                dim=-1,
            )
            + p
        )
        dr = torch.where(escaped, dr, power * r ** (power - 1.0) * dr + 1.0)
        z = torch.where(escaped[:, None], z, z_next)
    de = 0.5 * torch.log(r) * r / dr
    return de.reshape(points.shape[:-1])


def main() -> None:
    SCENES.mkdir(parents=True, exist_ok=True)

    # Hero: Mandelbulb from a raw PyTorch tensor field. Few fractal
    # iterations plus a light gaussian blur keep the surface detail above
    # the grid resolution, so the mesh stays smooth instead of noisy.
    grid = isoext.UniformGrid([192] * 3, aabb_min=[-1.2] * 3, aabb_max=[1.2] * 3)
    values = mandelbulb_field(grid.get_points(), iterations=6)
    grid.set_values(gaussian_smooth(values, sigma=1.0))
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
