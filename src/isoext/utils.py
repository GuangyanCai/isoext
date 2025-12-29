import torch


def write_obj(obj_path: str, v: torch.Tensor, f: torch.Tensor) -> None:
    """Write vertices and faces to an OBJ file.

    Args:
        obj_path: Path to the output OBJ file
        v: Tensor of vertices with shape (N, 3)
        f: Tensor of face indices with shape (M, 3)
    """
    with open(obj_path, "w") as obj_file:
        if v is None or f is None or v.numel() == 0 or f.numel() == 0:
            return
        v = v.tolist()
        f = (f + 1).tolist()

        lines = []
        for v0, v1, v2 in v:
            lines.append(f"v {v0} {v1} {v2}\n")

        for f0, f1, f2 in f:
            lines.append(f"f {f0} {f1} {f2}\n")

        obj_file.writelines(lines)


def make_grid(aabb: list[float], res: int | list[int], device: str = "cuda") -> torch.Tensor:
    """Create a uniform grid from an axis-aligned bounding box.

    Args:
        aabb: Axis-aligned bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        res: Resolution, either a single int or list of 3 ints for [x_res, y_res, z_res]
        device: Device to create the grid on (default: "cuda")

    Returns:
        Grid tensor with shape (x_res, y_res, z_res, 3)
    """
    if isinstance(res, int):
        res = [res] * 3

    s = [torch.linspace(aabb[i], aabb[i + 3], res[i], device=device) for i in range(3)]
    grid = torch.stack(torch.meshgrid(s, indexing="ij"), dim=-1)

    return grid
