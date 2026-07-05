#include "extraction.cuh"
#include "math.cuh"
#include "mt.cuh"

#include <thrust/device_vector.h>
#include <thrust/fill.h>

namespace {

// Each cell is split into 6 tetrahedra; up to 2 triangles per tetrahedron.
constexpr uint tets_per_cell = 6;
constexpr uint max_len = tets_per_cell * 2 * 3;

struct process_cell_op {
    float3 *v;
    const uint8_t *cases;
    const uint *cell_indices;
    const GridView view;
    const float level;

    process_cell_op(float3 *v, const uint8_t *cases, const uint *cell_indices,
                    const GridView &view, const float level)
        : v(v), cases(cases), cell_indices(cell_indices), view(view),
          level(level) {}

    __host__ __device__ void operator()(uint idx) {
        // The 6 Kuhn tetrahedra around the cell's main diagonal (corner 0
        // to corner 7, Morton corner order as documented in
        // shared_luts.cuh). The split is identical in every cell and the
        // face diagonals of neighboring cells coincide, so the mesh has no
        // cracks. Every tetrahedron is ordered with positive orientation,
        // which the triangle table relies on for outward-facing windings.
        const int tet_corners[6][4] = {
            {0, 4, 6, 7}, {0, 5, 4, 7}, {0, 6, 2, 7},
            {0, 2, 3, 7}, {0, 1, 5, 7}, {0, 3, 1, 7},
        };

        // Tetrahedron edges as pairs of local corners 0-3.
        const int tet_edges[6][2] = {
            {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
        };

        // Triangles per sign case, as triples of tetrahedron edge indices,
        // -1 padded. Bit i of the case is set when local corner i is
        // inside. Windings assume positive orientation and put the normal
        // on the outside; each complement case is its base case reversed.
        const int tet_tri_table[16][6] = {
            {-1, -1, -1, -1, -1, -1},   // 0000
            {0, 1, 2, -1, -1, -1},      // 0001
            {0, 4, 3, -1, -1, -1},      // 0010
            {1, 4, 3, 1, 2, 4},         // 0011
            {5, 1, 3, -1, -1, -1},      // 0100
            {0, 3, 5, 0, 5, 2},         // 0101
            {0, 4, 5, 0, 5, 1},         // 0110
            {5, 2, 4, -1, -1, -1},      // 0111
            {5, 4, 2, -1, -1, -1},      // 1000
            {0, 5, 4, 0, 1, 5},         // 1001
            {0, 5, 3, 0, 2, 5},         // 1010
            {5, 3, 1, -1, -1, -1},      // 1011
            {1, 3, 4, 1, 4, 2},         // 1100
            {0, 3, 4, -1, -1, -1},      // 1101
            {0, 2, 1, -1, -1, -1},      // 1110
            {-1, -1, -1, -1, -1, -1},   // 1111
        };

        uint cell_case = cases[idx];

        float3 c_p[8];
        float c_v[8];
        view.load_corners(cell_indices[idx], c_p, c_v);

        uint v_idx = idx * max_len;
        for (uint t = 0; t < tets_per_cell; t++) {
            const int *corners = tet_corners[t];
            uint tet_case = 0;
            for (int i = 0; i < 4; i++) {
                tet_case |= ((cell_case >> corners[i]) & 1) << i;
            }

            const int *tris = tet_tri_table[tet_case];
            for (int i = 0; i < 6 && tris[i] != -1; i += 3) {
                float3 tri[3];
                for (int j = 0; j < 3; j++) {
                    int c_0 = corners[tet_edges[tris[i + j]][0]];
                    int c_1 = corners[tet_edges[tris[i + j]][1]];
                    // Interpolate every edge in a canonical direction so
                    // that all cells produce bitwise identical vertices on
                    // shared edges, which vertex welding can then merge.
                    if (c_0 > c_1) {
                        int tmp = c_0;
                        c_0 = c_1;
                        c_1 = tmp;
                    }
                    float denom = c_v[c_1] - c_v[c_0];
                    float s =
                        (denom != 0.0f) ? (level - c_v[c_0]) / denom : 0.0f;
                    tri[j] = lerp(s, c_p[c_0], c_p[c_1]);
                }
                if (tri[0] != tri[1] && tri[0] != tri[2] && tri[1] != tri[2]) {
                    v[v_idx++] = tri[0];
                    v[v_idx++] = tri[1];
                    v[v_idx++] = tri[2];
                }
            }
        }
    }
};

}   // anonymous namespace

std::pair<NDArray<float3>, NDArray<int>>
marching_tetrahedra(Grid *grid, float level) {
    GridView view = grid->get_view();

    thrust::device_vector<uint8_t> cases =
        compute_cell_cases(view, grid->get_num_cells(), level);
    thrust::device_vector<uint> cell_indices = compact_active_cells(cases);
    uint num_cells = cell_indices.size();

    // Triangle soup with unused slots marked as NAN.
    thrust::device_vector<float3> v(num_cells * max_len);
    thrust::fill(v.begin(), v.end(), make_float3(NAN, NAN, NAN));

    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_cells),
                     process_cell_op(v.data().get(), cases.data().get(),
                                     cell_indices.data().get(), view, level));

    return soup_to_mesh(v);
}
