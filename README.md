# isoext

|      CI              | status |
|----------------------|--------|
| pip builds           | [![Pip Action Status][actions-pip-badge]][actions-pip-link] |
| wheels               | [![Wheel Action Status][actions-wheels-badge]][actions-wheels-link] |

[actions-pip-link]:        https://github.com/GuangyanCai/isoext/actions?query=workflow%3APip
[actions-pip-badge]:       https://github.com/GuangyanCai/isoext/workflows/Pip/badge.svg
[actions-wheels-link]:     https://github.com/GuangyanCai/isoext/actions?query=workflow%3AWheels
[actions-wheels-badge]:    https://github.com/GuangyanCai/isoext/workflows/Wheels/badge.svg

## Installation

1. Clone this repository
2. Run `pip install ./isoext`


## Quick Start

```python
import isoext
import torch 

def sphere_sdf(x):
    return x.norm(dim=-1) - 0.5

res = 128
x = torch.linspace(-1, 1, res)
y = torch.linspace(-1, 1, res)
z = torch.linspace(-1, 1, res)
grid = torch.stack(torch.meshgrid([x, y, z], indexing='xy'), dim=-1).cuda()
sdf = sphere_sdf(grid)

aabb = [-1, -1, -1, 1, 1, 1]
isolevel = -0.2

v, f = isoext.marching_cubes(sdf, aabb, isolevel)
```