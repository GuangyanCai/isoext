# Performance

All numbers are medians over repeated runs of `benchmarks/benchmark.py`,
measured on an NVIDIA GeForce RTX 5090 with CUDA 12.8, extracting a sphere
SDF at level 0. Timings include the full call from Python: case
classification, extraction, and vertex welding.

## Extraction time (median, ms)

| Grid    | Algorithm        | 128³  | 512³  |
|---------|------------------|-------|-------|
| uniform | marching_cubes   | 0.55  | 4.8   |
| uniform | get_intersection | 0.30  | 5.3   |
| uniform | dual_contouring  | 0.82  | 7.0   |
| sparse  | marching_cubes   | 0.44  | 1.4   |
| sparse  | get_intersection | 0.17  | 0.53  |
| sparse  | dual_contouring  | 0.78  | 2.3   |

Sparse grids only touch cells near the surface, so their cost scales with
the surface area rather than the volume. At 512³ that is 300k cells
instead of 134M.

## Reproducing

```bash
pixi run --environment cu128 bench

# or directly, with options:
pixi run --environment cu128 python benchmarks/benchmark.py \
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
