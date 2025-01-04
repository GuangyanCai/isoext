import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


def get_sdf_grad(sdf, p):
    p = p.requires_grad_()
    sdf_v = sdf(p)
    sdf_grad = torch.autograd.grad(sdf_v, p, grad_outputs=torch.ones_like(sdf_v))[0]
    return sdf_grad


def get_sdf_normal(sdf, p):
    sdf_grad = get_sdf_grad(sdf, p)
    return torch.nn.functional.normalize(sdf_grad, dim=-1)


class SDF(ABC):
    @abstractmethod
    def __call__(self, p):
        pass


@dataclass
class SphereSDF(SDF):
    radius: float

    def __call__(self, p):
        return p.norm(dim=-1) - self.radius


@dataclass
class TorusSDF(SDF):
    R: float
    r: float

    def __call__(self, p):
        tmp = p[..., [0, 1]].norm(dim=-1) - self.R
        return torch.stack([tmp, p[..., 2]], dim=-1).norm(dim=-1) - self.r


@dataclass
class CuboidSDF(SDF):
    size: List[float]  # full lengths in x, y, z directions

    def __call__(self, p):
        # Convert size to tensor and move to same device as input points
        # Divide by 2 since original formula uses half-lengths
        b = torch.tensor(self.size).to(p) / 2
        # Get distance from point to box boundary
        q = torch.abs(p) - b
        # Length of q.max(0) plus length of remaining positive components
        return torch.norm(
            torch.maximum(q, torch.zeros_like(q)), dim=-1
        ) + torch.minimum(q.max(dim=-1).values, torch.zeros_like(q[..., 0]))


@dataclass
class UnionOp(SDF):
    sdf_list: List[SDF]

    def __call__(self, p):
        results = [sdf(p) for sdf in self.sdf_list]
        return torch.stack(results, dim=-1).min(dim=-1).values


@dataclass
class SmoothUnionOp(SDF):
    sdf_list: List[SDF]
    k: float  # blending parameter

    def __call__(self, p):
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
    sdf_list: List[SDF]

    def __call__(self, p):
        results = [sdf(p) for sdf in self.sdf_list]
        return torch.stack(results, dim=-1).max(dim=-1).values


@dataclass
class NegationOp(SDF):
    sdf: SDF

    def __call__(self, p):
        return -self.sdf(p)


@dataclass
class TranslationOp(SDF):
    sdf: SDF
    offset: List[float]

    def __call__(self, p):
        return self.sdf(p - torch.tensor(self.offset).to(p))


@dataclass
class RotationOp(SDF):
    sdf: SDF
    axis: List[float]
    angle: float
    use_degree: bool = True

    def __post_init__(self):
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

        self.R = (
            cos_theta * torch.eye(3)
            + sin_theta * cpm
            + (1 - cos_theta) * (axis @ axis.T)
        )

    def __call__(self, p):
        return self.sdf(p @ self.R.to(p))
