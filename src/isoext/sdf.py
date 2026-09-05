from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import torch

from .isoext_ext import MeshBVH

__all__ = [
    "SDF",
    "SDFProtocol",
    "SphereSDF",
    "TorusSDF",
    "CuboidSDF",
    "MandelbulbSDF",
    "TriangleMeshSDF",
    "UnionOp",
    "IntersectionOp",
    "NegationOp",
    "SmoothUnionOp",
    "TranslationOp",
    "RotationOp",
    "get_sdf_grad",
    "get_sdf_normal",
    "project_to_surface",
]


class SDFProtocol(Protocol):
    """Protocol for SDF callable objects."""

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        """Evaluate SDF at given points.

        Args:
            p: Points tensor with shape (..., 3)

        Returns:
            SDF values tensor with shape (...)
        """
        ...


def get_sdf_grad(sdf: SDFProtocol, p: torch.Tensor) -> torch.Tensor:
    """Compute the gradient of an SDF at given points.

    Args:
        sdf: SDF function to evaluate
        p: Points tensor with shape (..., 3)

    Returns:
        Gradient tensor with shape (..., 3)
    """
    # Detach into a fresh leaf so the caller's tensor is not modified, and
    # enable grad mode so this also works inside torch.no_grad() blocks.
    p = p.detach().requires_grad_(True)
    with torch.enable_grad():
        sdf_v = sdf(p)
        sdf_grad = torch.autograd.grad(sdf_v, p, grad_outputs=torch.ones_like(sdf_v))[0]
    return sdf_grad


def project_to_surface(sdf: SDFProtocol, p: torch.Tensor, iters: int = 2) -> torch.Tensor:
    """Project points onto the zero level set of an SDF with Newton steps.

    Useful for refining the linearly interpolated intersection points from
    get_intersection before running dual contouring; more accurate points
    give sharper features.

    Args:
        sdf: SDF function to evaluate
        p: Points tensor with shape (..., 3)
        iters: Number of Newton steps

    Returns:
        Projected points tensor with shape (..., 3)
    """
    for _ in range(iters):
        p = p - sdf(p)[..., None] * get_sdf_normal(sdf, p)
    return p


def get_sdf_normal(sdf: SDFProtocol, p: torch.Tensor) -> torch.Tensor:
    """Compute normalized gradient (surface normal) of an SDF at given points.

    Args:
        sdf: SDF function to evaluate
        p: Points tensor with shape (..., 3)

    Returns:
        Normalized gradient tensor with shape (..., 3)
    """
    sdf_grad = get_sdf_grad(sdf, p)
    return torch.nn.functional.normalize(sdf_grad, dim=-1)


class SDF(ABC):
    """Abstract base class for Signed Distance Functions."""

    @abstractmethod
    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        """Evaluate SDF at given points.

        Args:
            p: Points tensor with shape (..., 3)

        Returns:
            SDF values tensor with shape (...)
        """
        pass


@dataclass
class SphereSDF(SDF):
    """SDF for a sphere centered at the origin."""

    radius: float

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        return p.norm(dim=-1) - self.radius


@dataclass
class TorusSDF(SDF):
    """SDF for a torus in the xy-plane.

    Args:
        R: Major radius (distance from center to tube center)
        r: Minor radius (tube radius)
    """

    R: float
    r: float

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        tmp = p[..., [0, 1]].norm(dim=-1) - self.R
        return torch.stack([tmp, p[..., 2]], dim=-1).norm(dim=-1) - self.r


@dataclass
class CuboidSDF(SDF):
    """SDF for an axis-aligned cuboid centered at the origin.

    Args:
        size: Full lengths in x, y, z directions
    """

    size: list[float]  # full lengths in x, y, z directions

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        # Convert size to tensor and move to same device as input points
        # Divide by 2 since original formula uses half-lengths
        b = torch.tensor(self.size, device=p.device, dtype=p.dtype) / 2
        # Get distance from point to box boundary
        q = torch.abs(p) - b
        # Length of q.max(0) plus length of remaining positive components
        q_max = q.max(dim=-1).values
        return torch.norm(torch.maximum(q, torch.zeros_like(q)), dim=-1) + torch.minimum(q_max, torch.zeros_like(q_max))


@dataclass
class MandelbulbSDF(SDF):
    """Distance estimator for the Mandelbulb fractal.

    The values estimate the distance to the fractal surface; they are not an
    exact SDF. Fewer iterations give a smoother, blobbier shape. The bulb
    fits inside a sphere of radius about 1.2.

    Args:
        power: Exponent of the iteration; 8 is the classic Mandelbulb.
        iterations: Number of fractal iterations.
    """

    power: float = 8.0
    iterations: int = 10

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        points = p.reshape(-1, 3)
        z = points.clone()
        dr = torch.ones(points.shape[0], device=points.device)
        r = z.norm(dim=-1)
        for _ in range(self.iterations):
            r = z.norm(dim=-1).clamp_min(1e-9)
            escaped = r > 2.0
            theta = torch.acos((z[:, 2] / r).clamp(-1.0, 1.0)) * self.power
            phi = torch.atan2(z[:, 1], z[:, 0]) * self.power
            zr = r**self.power
            z_next = (
                zr[:, None]
                * torch.stack(
                    [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()],
                    dim=-1,
                )
                + points
            )
            dr = torch.where(escaped, dr, self.power * r ** (self.power - 1.0) * dr + 1.0)
            z = torch.where(escaped[:, None], z, z_next)
        de = 0.5 * torch.log(r) * r / dr
        return de.reshape(p.shape[:-1])


class _MeshDistance(torch.autograd.Function):
    """Distance to a mesh, with the exact gradient of a distance field."""

    @staticmethod
    def forward(ctx, p, bvh, sign_method):
        flat = p.reshape(-1, 3).contiguous()
        dist, closest, _ = bvh.closest(flat)
        if sign_method == "winding":
            inside = bvh.winding_number(flat).abs() > 0.5
            sign = torch.where(inside, -1.0, 1.0)
        elif sign_method == "parity":
            sign = bvh.sign(flat)
        else:
            sign = torch.ones_like(dist)
        ctx.save_for_backward(flat, closest, dist, sign)
        return (sign * dist).reshape(p.shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        flat, closest, dist, sign = ctx.saved_tensors
        # d|p - q|/dp = (p - q) / |p - q|, undefined on the surface itself.
        direction = (flat - closest) / dist.clamp_min(1e-12)[:, None]
        grad = (grad_out.reshape(-1) * sign)[:, None] * direction
        return grad.reshape(grad_out.shape + (3,)), None, None


class TriangleMeshSDF(SDF):
    """Signed distance to a triangle mesh.

    The mesh is held on the GPU behind a bounding volume hierarchy, so the
    field can be evaluated at many points at once: sampling a mesh into a
    grid to extract it again, or building a field around scanned geometry.
    The gradient is the exact gradient of a distance field, so
    get_sdf_normal and project_to_surface work as for the analytic SDFs.

    Args:
        vertices: (V, 3) tensor of vertex positions on the CUDA device.
        faces: (F, 3) tensor of vertex indices.
        signed: Give points inside the mesh a negative distance. Pass False
            for the unsigned distance.
        sign: How the inside is decided. "winding" (default) uses the
            generalized winding number, which tolerates holes,
            self-intersections and disconnected pieces and does not depend
            on the mesh orientation. "parity" counts ray crossings, which
            is cheaper but needs a closed mesh.
    """

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor, signed: bool = True, sign: str = "winding"):
        if sign not in ("winding", "parity"):
            raise ValueError(f"sign must be 'winding' or 'parity', got {sign!r}")
        self.sign = sign if signed else None
        self._bvh = MeshBVH(vertices.detach().float().contiguous(), faces.detach().int().contiguous())

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        return _MeshDistance.apply(p, self._bvh, self.sign)

    def winding_number(self, p: torch.Tensor) -> torch.Tensor:
        """Generalized winding number at the given points.

        1 inside a closed mesh and 0 outside; fractional near holes and for
        triangle soups. Negative for an inward-oriented mesh.

        Args:
            p: Points tensor with shape (..., 3)

        Returns:
            Tensor with shape (...)
        """
        return self._bvh.winding_number(p.detach().reshape(-1, 3).float().contiguous()).reshape(p.shape[:-1])

    def closest_points(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project points onto the mesh.

        Args:
            p: Points tensor with shape (..., 3)

        Returns:
            A tuple (points, face_ids): the closest points on the mesh with
            shape (..., 3) and the index of the triangle holding each one
            with shape (...).
        """
        _, closest, tri = self._bvh.closest(p.detach().reshape(-1, 3).float().contiguous())
        return closest.reshape(p.shape), tri.reshape(p.shape[:-1])


@dataclass
class UnionOp(SDF):
    """Union operation combining multiple SDFs (minimum distance)."""

    sdf_list: list[SDF]

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        results = [sdf(p) for sdf in self.sdf_list]
        return torch.stack(results, dim=-1).min(dim=-1).values


@dataclass
class SmoothUnionOp(SDF):
    """Smooth union operation combining multiple SDFs with blending.

    Args:
        sdf_list: List of SDFs to combine
        k: Blending parameter (smaller values = sharper transition)
    """

    sdf_list: list[SDF]
    k: float  # blending parameter

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        results = torch.stack([sdf(p) for sdf in self.sdf_list], dim=-1)
        # Exponential smooth min, computed with logsumexp so that exp never
        # under- or overflows when |d| is much larger than k.
        return -self.k * torch.logsumexp(-results / self.k, dim=-1)


@dataclass
class IntersectionOp(SDF):
    """Intersection operation combining multiple SDFs (maximum distance)."""

    sdf_list: list[SDF]

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        results = [sdf(p) for sdf in self.sdf_list]
        return torch.stack(results, dim=-1).max(dim=-1).values


@dataclass
class NegationOp(SDF):
    """Negation operation (inverts SDF, creating inverse shape)."""

    sdf: SDF

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        return -self.sdf(p)


@dataclass
class TranslationOp(SDF):
    """Translation operation (moves SDF by an offset)."""

    sdf: SDF
    offset: list[float]

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        return self.sdf(p - torch.tensor(self.offset).to(p))


@dataclass
class RotationOp(SDF):
    """Rotation operation (rotates SDF around an axis).

    Args:
        sdf: SDF to rotate
        axis: Rotation axis as [x, y, z]
        angle: Rotation angle
        use_degree: If True, angle is in degrees; if False, in radians
    """

    sdf: SDF
    axis: list[float]
    angle: float
    use_degree: bool = True

    def __post_init__(self) -> None:
        axis = torch.tensor(self.axis).float()
        axis = torch.nn.functional.normalize(axis, dim=0).reshape(3, 1)

        angle = torch.tensor(self.angle).float()
        if self.use_degree:
            angle = torch.deg2rad(angle)

        sin_theta = torch.sin(angle)
        cos_theta = torch.cos(angle)

        cpm = torch.zeros((3, 3))
        cpm[0, 1] = -axis[2]
        cpm[0, 2] = axis[1]
        cpm[1, 0] = axis[2]
        cpm[1, 2] = -axis[0]
        cpm[2, 0] = -axis[1]
        cpm[2, 1] = axis[0]

        self.R = cos_theta * torch.eye(3) + sin_theta * cpm + (1 - cos_theta) * (axis @ axis.T)

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        return self.sdf(p @ self.R.to(p))
