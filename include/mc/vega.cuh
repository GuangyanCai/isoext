#pragma once

#include "mc/base.cuh"

namespace mc {

// Topologically correct marching cubes after Chernyaev's MC33, using the
// table and decision procedure of Vega et al., "A Fast and Memory-Saving
// Marching Cubes 33 Implementation with the Correct Interior Test" (2019).
// Unlike the Lewiner variant, the interior test analyzes the trilinear
// interpolant correctly, so the extracted topology is invariant under
// reflections of the input field.
class Vega : public MCBase {
  public:
    static constexpr size_t max_triangles = 12;

    void run(float3 *v, const uint num_cells, const uint8_t *cases,
             const uint *cell_indices, const GridView &view,
             const float level) override;

    size_t get_max_triangles() const override { return max_triangles; }
};

}   // namespace mc
