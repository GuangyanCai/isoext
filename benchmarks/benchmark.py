"""Benchmark isoext extraction algorithms.

Usage:
    pixi run bench
    pixi run python benchmarks/benchmark.py \
        --res 32 64 --iters 50 --json benchmarks/results/out.json
"""

import argparse
import json
import statistics
import time
from importlib.metadata import version

import torch

import isoext
from isoext.sdf import SphereSDF


def time_fn(fn, iters, warmup=10):
    """Time a GPU function, returning millisecond percentiles.

    The very first invocation is reported separately: with PTX-only builds it
    includes the driver's JIT compilation (cached on disk afterwards).
    """
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first_call = (time.perf_counter() - t0) * 1e3

    for _ in range(warmup - 1):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "p10_ms": times[int(len(times) * 0.10)],
        "p90_ms": times[int(len(times) * 0.90)],
        "first_call_ms": first_call,
        "iters": iters,
    }


def make_uniform_grid(sdf, res):
    grid = isoext.UniformGrid([res] * 3, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    grid.set_values(sdf(grid.get_points()))
    return grid


def make_sparse_grid(sdf, res, chunk_size=2_000_000):
    grid = isoext.SparseGrid([res] * 3, aabb_min=[-1, -1, -1], aabb_max=[1, 1, 1])
    for chunk in grid.get_potential_cell_indices(chunk_size):
        points = grid.get_points_by_cell_indices(chunk)
        filtered = grid.filter_cell_indices(chunk, sdf(points), level=0.0)
        if len(filtered) > 0:
            grid.add_cells(filtered)
    if grid.get_num_cells() > 0:
        grid.set_values(sdf(grid.get_points()))
    return grid


def benchmark_grid(grid, label, iters, results):
    cases = {
        "mc_nagae": lambda: isoext.marching_cubes(grid, method="nagae"),
        "mc_lorensen": lambda: isoext.marching_cubes(grid, method="lorensen"),
        "mc_lewiner": lambda: isoext.marching_cubes(grid, method="lewiner"),
        "mc_vega": lambda: isoext.marching_cubes(grid, method="vega"),
        "marching_tets": lambda: isoext.marching_tetrahedra(grid),
        "get_intersection": lambda: isoext.get_intersection(grid, compute_normals=True),
        "dc_ju": lambda: isoext.dual_contouring(grid),
        "dc_carrera": lambda: isoext.dual_contouring(grid, method="carrera"),
        "dual_marching_cubes": lambda: isoext.dual_marching_cubes(grid),
        "surface_nets": lambda: isoext.surface_nets(grid),
    }
    for algo, fn in cases.items():
        try:
            stats = time_fn(fn, iters)
            status = f"{stats['median_ms']:9.3f} ms   (first {stats['first_call_ms']:8.3f} ms)"
        except Exception as e:  # noqa: BLE001 - record and continue
            stats = {"error": f"{type(e).__name__}: {e}"}
            status = "   failed"
        results.append({"case": label, "algo": algo, **stats})
        print(f"  {label:24s} {algo:18s} {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sparse-max-res", type=int, default=256)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    sdf = SphereSDF(radius=0.5)
    results = []

    print(f"isoext {version('isoext')} | {torch.cuda.get_device_name()} | CUDA {torch.version.cuda}")
    print(f"{'':2s}{'case':24s} {'algorithm':18s} {'median':>9s}")

    for res in args.res:
        # Fewer iterations for the big grids to keep runtime reasonable.
        iters = max(10, args.iters // max(1, res // 64))

        grid = make_uniform_grid(sdf, res)
        benchmark_grid(grid, f"uniform_{res}", iters, results)
        del grid

        if res <= args.sparse_max_res:
            grid = make_sparse_grid(sdf, res)
            benchmark_grid(grid, f"sparse_{res}", iters, results)
            del grid

    if args.json:
        meta = {
            "isoext": version("isoext"),
            "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        }
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
