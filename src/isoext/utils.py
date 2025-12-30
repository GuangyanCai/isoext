import torch
import torch.nn.functional as F


def gaussian_smooth(
    field: torch.Tensor,
    sigma: float = 1.0,
    kernel_size: int | None = None,
) -> torch.Tensor:
    """Smooth a 3D scalar field using a Gaussian filter.

    Args:
        field: Input scalar field with shape (X, Y, Z)
        sigma: Standard deviation of the Gaussian kernel (default: 1.0)
        kernel_size: Size of the kernel. If None, uses ceil(6 * sigma) | 1 to ensure odd size.

    Returns:
        Smoothed scalar field with the same shape as input
    """
    if kernel_size is None:
        kernel_size = int(6 * sigma) | 1  # Ensure odd

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, device=field.device, dtype=field.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 3D kernel via outer products
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    kernel_3d = kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)

    # Apply conv3d with padding to preserve size
    padding = kernel_size // 2
    field_5d = field.view(1, 1, *field.shape)
    # Use replicate padding for better boundary handling
    field_5d = F.pad(field_5d, [padding] * 6, mode="replicate")
    smoothed = F.conv3d(field_5d, kernel_3d)

    return smoothed.reshape(field.shape)


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
