"""Well-known test meshes, downloaded on first use.

The meshes are not shipped with the package. The Stanford models may be
used and redistributed for research but not commercially, so they are
fetched from the Stanford Computer Graphics Laboratory when first
requested and cached locally; Spot is public domain.
"""

import gzip
import hashlib
import io
import os
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = ["ASSETS", "load_mesh", "cache_dir"]


@dataclass(frozen=True)
class Asset:
    url: str
    sha256: str
    member: str | None  # file inside the archive, or None for a single gzip file
    source: str
    closed: bool  # watertight, so a signed distance is meaningful


ASSETS = {
    "bunny": Asset(
        url="http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
        sha256="a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84",
        member="bunny/reconstruction/bun_zipper.ply",
        source="Stanford Computer Graphics Laboratory",
        closed=False,
    ),
    "armadillo": Asset(
        url="http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz",
        sha256="8b9b56cc36e66d54429b1e1e75bd89e833645bfe0dc7c1afd1205877a7356a3f",
        member=None,
        source="Stanford Computer Graphics Laboratory",
        closed=True,
    ),
    "dragon": Asset(
        url="http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz",
        sha256="74ac1d90989c9b1732edee82d57e9ce71452144cf4355f108d8c9c616d28d02f",
        member="dragon_recon/dragon_vrip.ply",
        source="Stanford Computer Graphics Laboratory",
        closed=False,
    ),
    "spot": Asset(
        url="https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/spot.zip",
        sha256="1d4082da4e940f6a4cfdf648fd77e1647ac613c618dbff933f1ba23fad254608",
        member="spot/spot_quadrangulated.obj",
        source="Keenan Crane (public domain)",
        closed=True,
    ),
}


def cache_dir() -> Path:
    """Directory holding the downloaded meshes.

    ``$ISOEXT_CACHE`` if set, else ``$XDG_CACHE_HOME/isoext`` or
    ``~/.cache/isoext``.
    """
    if "ISOEXT_CACHE" in os.environ:
        return Path(os.environ["ISOEXT_CACHE"])
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "isoext"


def load_mesh(name: str, size: float = 1.6, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Load one of the test meshes as (vertices, faces) tensors.

    The mesh is downloaded and cached on first use, then centered at the
    origin, scaled so that its longest side has length ``size`` (so it fits
    the default grid domain [-1, 1] with a margin) and turned z-up.

    Args:
        name: One of the keys of ASSETS: "bunny", "armadillo", "dragon"
            or "spot". "armadillo" and "spot" are closed; "bunny" and
            "dragon" have holes at the bottom.
        size: Length of the longest side of the bounding box after scaling.
        device: Device of the returned tensors.

    Returns:
        A tuple (vertices, faces): an (N, 3) float32 tensor and an (M, 3)
        int32 tensor of triangles.
    """
    if name not in ASSETS:
        raise KeyError(f"Unknown asset {name!r}; available: {sorted(ASSETS)}")
    asset = ASSETS[name]
    npz = cache_dir() / f"{name}.npz"
    if not npz.exists():
        data = _fetch(asset)
        vertices, faces = _parse(asset, data)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, vertices=vertices, faces=faces)
    else:
        loaded = np.load(npz)
        vertices, faces = loaded["vertices"], loaded["faces"]

    # Center, scale and rotate from y-up to z-up.
    vertices = vertices - (vertices.min(0) + vertices.max(0)) / 2
    vertices = vertices * (size / (vertices.max(0) - vertices.min(0)).max())
    vertices = np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], axis=1)
    return (
        torch.from_numpy(np.ascontiguousarray(vertices, dtype=np.float32)).to(device),
        torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int32)).to(device),
    )


def _fetch(asset: Asset) -> bytes:
    """The archive's bytes, from the cache or downloaded and verified."""
    path = cache_dir() / Path(asset.url).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(asset.url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            raise RuntimeError(f"Checksum mismatch for {asset.url}; the file may have changed upstream.")
        path.write_bytes(data)
    return path.read_bytes()


def _parse(asset: Asset, data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Extract the mesh file from the archive and parse it."""
    name = Path(asset.url).name
    if name.endswith(".tar.gz"):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            content = tar.extractfile(asset.member).read()
    elif name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            content = archive.read(asset.member)
    else:
        content = gzip.decompress(data)
    member = asset.member or name.removesuffix(".gz")
    return _read_obj(content) if member.endswith(".obj") else _read_ply(content)


def _triangulate(polygons: np.ndarray) -> np.ndarray:
    """Fan-triangulate (M, k) polygons into (M * (k - 2), 3) triangles."""
    k = polygons.shape[1]
    return np.concatenate([polygons[:, [0, i, i + 1]] for i in range(1, k - 1)])


def _read_obj(content: bytes) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    for line in content.decode().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            faces.append([int(p.split("/")[0]) - 1 for p in parts[1:]])
    vertices = np.array(vertices, dtype=np.float32)
    counts = {len(f) for f in faces}
    tris = [np.array([f for f in faces if len(f) == k], dtype=np.int64) for k in sorted(counts)]
    return vertices, np.concatenate([_triangulate(t) for t in tris]).astype(np.int32)


_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}  # fmt: skip


def _read_ply(content: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Read the vertex positions and faces of an ascii or binary PLY file.

    Handles extra scalar properties on vertices and faces; faces are
    expected to be polygons with a single index list.
    """
    header_end = content.index(b"end_header\n") + len(b"end_header\n")
    header = content[:header_end].decode().splitlines()
    body = content[header_end:]

    fmt = None
    elements = []  # (name, count, [(prop_name, dtype or ("list", count_dtype, item_dtype))])
    for line in header:
        parts = line.split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if parts[1] == "list":
                elements[-1][2].append((parts[4], ("list", _PLY_TYPES[parts[2]], _PLY_TYPES[parts[3]])))
            else:
                elements[-1][2].append((parts[2], _PLY_TYPES[parts[1]]))

    vertices = faces = None
    if fmt == "ascii":
        tokens = body.split()
        pos = 0
        for name, count, props in elements:
            if any(isinstance(t, tuple) for _, t in props):
                rows = []
                for _ in range(count):
                    row = []
                    for _, t in props:
                        if isinstance(t, tuple):
                            n = int(tokens[pos])
                            row.append([int(x) for x in tokens[pos + 1 : pos + 1 + n]])
                            pos += 1 + n
                        else:
                            row.append(tokens[pos])
                            pos += 1
                    rows.append(row)
                if name == "face":
                    idx = [i for i, (_, t) in enumerate(props) if isinstance(t, tuple)][0]
                    polys = [r[idx] for r in rows]
                    faces = _polygons_to_triangles(polys)
            else:
                width = len(props)
                block = np.array(tokens[pos : pos + count * width], dtype=np.float64).reshape(count, width)
                pos += count * width
                if name == "vertex":
                    cols = [i for i, (p, _) in enumerate(props) if p in ("x", "y", "z")]
                    vertices = block[:, cols].astype(np.float32)
    else:
        endian = ">" if fmt == "binary_big_endian" else "<"
        offset = 0
        for name, count, props in elements:
            if any(isinstance(t, tuple) for _, t in props):
                # Every polygon is assumed to have the vertex count of the
                # first one (checked below), which makes the block a
                # fixed-size record array.
                list_i = next(i for i, (_, t) in enumerate(props) if isinstance(t, tuple))
                list_name, (_, count_type, item_type) = props[list_i]
                pre = sum(np.dtype(t).itemsize for _, t in props[:list_i])
                n = int(np.frombuffer(body, dtype=endian + count_type, count=1, offset=offset + pre)[0])
                fields = [
                    (p, [("n", endian + count_type), ("idx", endian + item_type, (n,))])
                    if isinstance(t, tuple)
                    else (p, endian + t)
                    for p, t in props
                ]
                dtype = np.dtype(fields)
                block = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
                offset += count * dtype.itemsize
                if name == "face":
                    if not (block[list_name]["n"] == n).all():
                        raise ValueError("PLY faces with mixed vertex counts are not supported in binary files")
                    faces = _triangulate(block[list_name]["idx"].astype(np.int64))
            else:
                dtype = np.dtype([(p, endian + t) for p, t in props])
                block = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
                offset += count * dtype.itemsize
                if name == "vertex":
                    vertices = np.stack([block["x"], block["y"], block["z"]], axis=1).astype(np.float32)
    if vertices is None or faces is None:
        raise ValueError("PLY file without vertex or face elements")
    return vertices, faces.astype(np.int32)


def _polygons_to_triangles(polys: list[list[int]]) -> np.ndarray:
    counts = {len(p) for p in polys}
    parts = [np.array([p for p in polys if len(p) == k], dtype=np.int64) for k in sorted(counts)]
    return np.concatenate([_triangulate(p) for p in parts])
