import torch 

def write_obj(obj_path, v, f):
    with open(obj_path, 'w') as obj_file:
        if v is None or f is None or v.numel() == 0 or f.numel() == 0:
            return

        v = v.tolist()
        f = (f + 1).tolist()

        lines = []
        for v0, v1, v2 in v:
            lines.append(f'v {v0} {v1} {v2}\n')

        for f0, f1, f2 in f:
            lines.append(f'f {f0} {f1} {f2}\n')

        obj_file.writelines(lines)

def make_grid(aabb, res, device='cuda'):
    if isinstance(res, int):
        res = [res] * 3

    s = [torch.linspace(aabb[i], aabb[i + 3], res[i], device=device) for i in range(3)]
    grid = torch.stack(torch.meshgrid(s, indexing='ij'), dim=-1)

    return grid