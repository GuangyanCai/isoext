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
the surface area rather than the volume — at 512³ that is the difference
between processing 134M cells and 300k cells.

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

- **First call**: the shipped extension contains PTX that the driver
  JIT-compiles for your GPU on first use (a couple of seconds, cached on
  disk afterwards). See {doc}`installation` for details. The benchmark
  excludes this by warming up before timing.
- **Memory**: cell corner positions and indices are computed on the fly
  inside the kernels, so uniform-grid extraction allocates memory
  proportional to the extracted surface, not the grid volume — 1024³ dense
  grids fit comfortably on a consumer GPU.
- Timings vary with GPU, driver, and the surface complexity of your field;
  treat these as orders of magnitude, not guarantees.
