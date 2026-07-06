# Performance

All numbers are medians over repeated runs of `benchmarks/benchmark.py`,
measured on an NVIDIA GeForce RTX 5090 with CUDA 12.8, extracting a sphere
SDF at level 0. Timings include the full call from Python: case
classification, extraction, and vertex welding.

## Extraction time (median, ms)

| Grid    | Algorithm           | 128³  | 512³  |
|---------|---------------------|-------|-------|
| uniform | marching_cubes      | 0.55  | 5.4   |
| uniform | marching_tetrahedra | 0.68  | 6.7   |
| uniform | surface_nets        | 0.81  | 6.9   |
| uniform | dual_contouring     | 0.83  | 7.0   |
| uniform | get_intersection    | 0.26  | 5.3   |
| sparse  | marching_cubes      | 0.48  | 1.9   |
| sparse  | marching_tetrahedra | 0.61  | 3.2   |
| sparse  | surface_nets        | 0.73  | 2.2   |
| sparse  | dual_contouring     | 0.75  | 2.3   |
| sparse  | get_intersection    | 0.13  | 0.53  |

Sparse grids only touch cells near the surface, so their cost scales with
the surface area rather than the volume. At 512³ that is 300k cells
instead of 134M.

## Marching cubes variants (median, ms)

| Variant  | uniform 512³ | sparse 512³ |
|----------|--------------|-------------|
| lorensen | 4.9          | 1.4         |
| nagae    | 5.0          | 1.5         |
| lewiner  | 5.5          | 2.0         |
| vega     | 5.4          | 1.9         |

## Reproducing

The environment setup is described in {doc}`development`. Then:

```bash
pixi run bench

# or directly, with options:
pixi run python benchmarks/benchmark.py \
    --res 32 64 128 256 512 --json results.json
```

The harness reports the median, 10th and 90th percentiles, and the first
call separately.

## Notes

- The first call after installation is slow (a few seconds): the driver
  compiles the shipped PTX for your GPU and caches the result. See
  {doc}`installation`. The benchmark warms up before timing, so the table
  excludes this.
- The kernels compute cell corners on the fly instead of storing them, so
  memory use grows with the extracted surface, not the grid volume. A
  dense 1024³ grid fits on a consumer GPU.
- Numbers depend on the GPU, driver, and field. Expect different results
  on other machines.
