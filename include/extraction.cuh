#pragma once

#include "grid/grid.cuh"
#include "ndarray.cuh"

#include <thrust/device_vector.h>

#include <utility>

// Building blocks shared by the per-cell extraction methods (the marching
// cubes variants and marching tetrahedra). A method combines them as:
//
//   1. compute_cell_cases: the corner-sign byte of every cell.
//   2. compact_active_cells: drop the cells the surface does not cross.
//   3. Run a kernel that writes triangles for each active cell into a
//      buffer pre-filled with NAN; unused slots stay NAN.
//   4. soup_to_mesh: drop the NAN slots and weld the triangle soup into an
//      indexed mesh.
//
// Custom methods can reuse any subset of these.

// Corner-sign byte per cell; bit i is set when corner i lies below level.
thrust::device_vector<uint8_t>
compute_cell_cases(const GridView &view, uint num_cells, float level);

// Remove cells whose corners are all inside or all outside (case 0 or 255)
// from cases, and return the grid cell index of each surviving cell.
thrust::device_vector<uint>
compact_active_cells(thrust::device_vector<uint8_t> &cases);

// Compact the NAN-marked triangle soup and weld it into an indexed mesh.
std::pair<NDArray<float3>, NDArray<int>>
soup_to_mesh(thrust::device_vector<float3> &v);
