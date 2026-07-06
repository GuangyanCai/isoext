#pragma once

#include "mc/base.cuh"

namespace mc {

// Topologically correct marching cubes after Chernyaev's MC33, using the
// tables and decision procedure of Lewiner et al., "Efficient
// implementation of Marching Cubes' cases with topological guarantees"
// (2003). Ambiguous faces are resolved with bilinear saddle tests and
// interior ambiguities with tests on the trilinear interpolant, so the
// mesh follows the topology of the trilinear interpolant of the samples.
class Lewiner : public MCBase {
  public:
    static constexpr size_t max_triangles = 12;

    void run(float3 *v, const uint num_cells, const uint8_t *cases,
             const uint *cell_indices, const GridView &view,
             const float level) override;

    size_t get_max_triangles() const override { return max_triangles; }
};

}   // namespace mc
