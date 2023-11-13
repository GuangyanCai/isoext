
# isoext: Isosurface Extraction on GPU

## Overview

Welcome to `isoext` — a Python library designed for efficient isosurface extraction, leveraging the power of GPU computing and comes with `pytorch` support. Our library attempts to implement a collection of classic isosurface extraction algorithms. Currently, only the following algorithms are supported, but more will come in the future:
* Marching cubes
  * `lorensen`: the original marching cubes algorithm from the paper [Marching cubes: A high resolution 3D surface construction algorithm](https://dl.acm.org/doi/10.1145/37402.37422).

|      CI              | status |
|----------------------|--------|
| pip builds           | [![Pip Action Status][actions-pip-badge]][actions-pip-link] |
| wheels               | [![Wheel Action Status][actions-wheels-badge]][actions-wheels-link] |

[actions-pip-link]:        https://github.com/GuangyanCai/isoext/actions?query=workflow%3APip
[actions-pip-badge]:       https://github.com/GuangyanCai/isoext/workflows/Pip/badge.svg
[actions-wheels-link]:     https://github.com/GuangyanCai/isoext/actions?query=workflow%3AWheels
[actions-wheels-badge]:    https://github.com/GuangyanCai/isoext/workflows/Wheels/badge.svg

## Installation

To install `isoext`, simply run:

```bash
pip install isoext
```

## Quick Start

Here's a simple example to get you started:

```python
import isoext
import torch 

def sphere_sdf(x):
    return x.norm(dim=-1) - 0.5

res = 128
x = torch.linspace(-1, 1, res)
y = torch.linspace(-1, 1, res)
z = torch.linspace(-1, 1, res)
grid = torch.stack(torch.meshgrid([x, y, z], indexing='xy'), dim=-1)
sdf = sphere_sdf(grid).cuda() # Only accept a gpu tensor from pytorch for now

aabb = [-1, -1, -1, 1, 1, 1]
isolevel = -0.2

v, f = isoext.marching_cubes(sdf, aabb, isolevel)
```

## Task List
- [ ] Fix docstring.
- [ ] Implement MC33.
- [ ] Add `numpy` support.

## License

`isoext` is released under the [MIT License](LICENSE). Feel free to use it in your projects.

## Acknowledgments
* We use [Thrust](https://developer.nvidia.com/thrust) for GPU computing and [nanobind](https://github.com/wjakob/nanobind) for Python binding. 
* The LUTs for `lorensen` are borrowed from Paul Bourke (https://paulbourke.net/geometry/polygonise/).