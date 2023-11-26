import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

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
class UnionOp(SDF):
    sdf_list: List[SDF]

    def __call__(self, p):
        results = [sdf(p) for sdf in self.sdf_list]
        return torch.stack(results, dim=-1).min(dim=-1).values

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
        cpm[0, 2] =  axis[1]
        cpm[1, 0] =  axis[2]
        cpm[1, 2] = -axis[0]
        cpm[2, 0] = -axis[1]
        cpm[2, 1] =  axis[0]

        self.R = cos_theta * torch.eye(3) + sin_theta * cpm + (1 - cos_theta) * (axis @ axis.T)

    def __call__(self, p):
        return self.sdf(p @ self.R.to(p))
        