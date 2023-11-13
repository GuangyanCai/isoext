import isoext
import torch 

def test_basic():

    def sphere_sdf(x):
        return x.norm(dim=-1) - 0.5

    res = 3
    x = torch.linspace(-1, 1, res)
    y = torch.linspace(-1, 1, res)
    z = torch.linspace(-1, 1, res)
    grid = torch.stack(torch.meshgrid([x, y, z], indexing='xy'), dim=-1).cuda()
    sdf = sphere_sdf(grid)

    v, f = isoext.marching_cubes(sdf, [-1, -1, -1, 1, 1, 1], 0)

    assert v.shape == torch.Size([6, 3])
    assert f.shape == torch.Size([8, 3])
