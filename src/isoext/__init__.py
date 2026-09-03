import importlib.util

if importlib.util.find_spec("torch") is None:
    raise ImportError("PyTorch is required but not installed. Please install PyTorch with CUDA support.\n")

from importlib.metadata import version as _version

from . import sdf
from .dc import dual_contouring
from .isoext_ext import (
    Intersection,
    SparseGrid,
    UniformGrid,
    dual_marching_cubes,
    get_intersection,
    marching_cubes,
    marching_tetrahedra,
    surface_nets,
)
from .utils import gaussian_smooth, write_obj

__version__ = _version("isoext")

__all__ = [
    "Intersection",
    "SparseGrid",
    "UniformGrid",
    "dual_contouring",
    "dual_marching_cubes",
    "gaussian_smooth",
    "get_intersection",
    "marching_cubes",
    "marching_tetrahedra",
    "sdf",
    "surface_nets",
    "write_obj",
]
