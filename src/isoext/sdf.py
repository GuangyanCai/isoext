from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import torch


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
    p = p.requires_grad_()
    sdf_v = sdf(p)
    sdf_grad = torch.autograd.grad(sdf_v, p, grad_outputs=torch.ones_like(sdf_v))[0]
    return sdf_grad


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
        results = [sdf(p) for sdf in self.sdf_list]
        # results is [d1, d2, ...]
        # We take a pairwise or reduce approach to do a smooth union:
        d = results[0]
        for i in range(1, len(results)):
            d2 = results[i]
            d = -self.k * torch.log(torch.exp(-d / self.k) + torch.exp(-d2 / self.k))
        return d


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
