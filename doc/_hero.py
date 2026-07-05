"""Generate the interactive scenes embedded on the landing page.

Run manually after changing the shapes:

    pixi run --environment doc python _hero.py
"""

import time
from pathlib import Path

import torch

import isoext
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


def menger_field(points: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    """SDF of a Menger sponge (box with iteratively drilled cross holes)."""
    q = points.abs() - 1.0
    d = q.clamp_min(0.0).norm(dim=-1) + q.amax(dim=-1).clamp_max(0.0)

    s = 1.0
    for _ in range(iterations):
        a = torch.remainder(points * s, 2.0) - 1.0
        s *= 3.0
        r = (1.0 - 3.0 * a.abs()).abs()
        da = torch.maximum(r[..., 0], r[..., 1])
        db = torch.maximum(r[..., 1], r[..., 2])
        dc = torch.maximum(r[..., 2], r[..., 0])
        cross = (torch.minimum(da, torch.minimum(db, dc)) - 1.0) / s
        d = torch.maximum(d, cross)
    return d


def main() -> None:
    SCENES.mkdir(parents=True, exist_ok=True)

    # Hero: Mandelbulb from a raw PyTorch tensor field.
    grid = isoext.UniformGrid([160] * 3, aabb_min=[-1.2] * 3, aabb_max=[1.2] * 3)
    grid.set_values(mandelbulb_field(grid.get_points()))
    torch.cuda.synchronize()
    start = time.perf_counter()
    v, f = isoext.marching_cubes(grid)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    print(f"mandelbulb: {v.shape[0]:,} vertices, extracted in {elapsed_ms:.1f} ms")
    save_scene(SCENES / "hero.viser", v, f, color=(0.88, 0.45, 0.34))

    # Comparison strip: the same Menger sponge field, meshed by marching
    # cubes and by dual contouring (which preserves the sharp edges).
    grid = isoext.UniformGrid([81] * 3, aabb_min=[-1.1] * 3, aabb_max=[1.1] * 3)
    grid.set_values(menger_field(grid.get_points()))
    v, f = isoext.marching_cubes(grid)
    print(f"menger mc: {v.shape[0]:,} vertices")
    save_scene(SCENES / "menger_mc.viser", v, f, color=(0.64, 0.68, 0.78), flat_shading=True)
    v, f = isoext.dual_contouring(grid)
    print(f"menger dc: {v.shape[0]:,} vertices")
    save_scene(SCENES / "menger_dc.viser", v, f, color=(0.64, 0.68, 0.78), flat_shading=True)

    for name in ["hero.viser", "menger_mc.viser", "menger_dc.viser"]:
        size = (SCENES / name).stat().st_size
        print(f"{name}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
