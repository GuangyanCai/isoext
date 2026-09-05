#include "mc/lewiner.cuh"
#include "mc/lewiner_luts.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

// Marching cubes with topological guarantees, after Chernyaev's MC33.
//
// This is a port of Lewiner et al., "Efficient implementation of Marching
// Cubes' cases with topological guarantees" (Journal of Graphics Tools,
// 2003), as distributed with scikit-image, including its handling of
// borderline test outcomes so the two implementations can be compared
// directly. The method itself is from Chernyaev, "Marching Cubes 33:
// Construction of Topologically Correct Isosurfaces" (1995).
//
// Known limitations, inherited from the reference implementation and
// documented by Custodio et al., "Practical considerations on Marching
// Cubes 33 topological correctness" (Computers & Graphics, 2013):
//
//  - The interior test evaluates the trilinear interpolant on a single
//    section instead of analyzing it fully, so it can misjudge tunnel
//    connectivity in rare configurations, and it is not invariant under
//    reflections: mirroring the input field can change the local topology.
//  - Parts of the case 13 handling were later corrected by Custodio et
//    al.; those corrections are not included here.
//
// The meshes are always watertight and crack free; the limitations only
// affect which of the locally plausible topologies is chosen. The "vega"
// variant implements the corrected interior test and avoids them.
//
// Cells are processed in Lewiner's corner numbering and converted to ours
// only when vertices are emitted.

namespace mc {

namespace {
static MCRegistrar<Lewiner> registrar("lewiner");

constexpr float eps = 1.19209290e-07f;   // FLT_EPSILON

// Lewiner corner i sits at our Morton corner our_corner[i].
__host__ __device__ inline int
our_corner(int lewiner) {
    const int ours[8] = {0, 4, 6, 2, 1, 5, 7, 3};
    return ours[lewiner];
}

// Per-thread state and logic for one cell.
struct cell_worker {
    float3 *v;
    const signed char *luts;
    uint v_idx;
    float lv[8];    // corner values relative to the level, Lewiner order
    float3 lp[8];   // corner positions, Lewiner order

    // Sign of the bilinear interpolant at the saddle point of a face.
    __host__ __device__ bool face_test(int face) {
        int abs_face = face < 0 ? -face : face;
        float a, b, c, d;
        switch (abs_face) {
        case 1:
            a = lv[0];
            b = lv[4];
            c = lv[5];
            d = lv[1];
            break;
        case 2:
            a = lv[1];
            b = lv[5];
            c = lv[6];
            d = lv[2];
            break;
        case 3:
            a = lv[2];
            b = lv[6];
            c = lv[7];
            d = lv[3];
            break;
        case 4:
            a = lv[3];
            b = lv[7];
            c = lv[4];
            d = lv[0];
            break;
        case 5:
            a = lv[0];
            b = lv[3];
            c = lv[2];
            d = lv[1];
            break;
        default:
            a = lv[4];
            b = lv[7];
            c = lv[6];
            d = lv[5];
            break;
        }
        float ac_bd = a * c - b * d;
        if (ac_bd > -eps && ac_bd < eps) {
            return face >= 0;
        }
        return face * a * ac_bd >= 0.0f;
    }

    // Sign of the trilinear interpolant on the section through a reference
    // edge, deciding whether opposite sheets connect inside the cell.
    __host__ __device__ bool interior_test(int case_num, int config, int sub,
                                           int s) {
        float at = 0.0f, bt = 0.0f, ct = 0.0f, dt = 0.0f;
        float t;

        if (case_num == 4 || case_num == 10) {
            float a = (lv[4] - lv[0]) * (lv[6] - lv[2]) -
                      (lv[7] - lv[3]) * (lv[5] - lv[1]);
            float b = lv[2] * (lv[4] - lv[0]) + lv[0] * (lv[6] - lv[2]) -
                      lv[1] * (lv[7] - lv[3]) - lv[3] * (lv[5] - lv[1]);
            t = -b / (2.0f * a + eps);
            if (t < 0.0f || t > 1.0f) {
                return s > 0;
            }
            at = lv[0] + (lv[4] - lv[0]) * t;
            bt = lv[3] + (lv[7] - lv[3]) * t;
            ct = lv[2] + (lv[6] - lv[2]) * t;
            dt = lv[1] + (lv[5] - lv[1]) * t;
        } else {
            int edge;
            if (case_num == 6) {
                edge = mc33::TEST6(luts, config, 2);
            } else if (case_num == 7) {
                edge = mc33::TEST7(luts, config, 4);
            } else if (case_num == 12) {
                edge = mc33::TEST12(luts, config, 3);
            } else {   // case 13
                edge = mc33::TILING13_5_1(luts, config, sub, 0);
            }

            // Interpolate the crossing on the reference edge, then sample
            // the three parallel cube edges at the same parameter. Rows
            // follow the reference implementation edge by edge.
            const int rows[12][4][2] = {
                {{0, 1}, {3, 2}, {7, 6}, {4, 5}},
                {{1, 2}, {0, 3}, {4, 7}, {5, 6}},
                {{2, 3}, {1, 0}, {5, 4}, {6, 7}},
                {{3, 0}, {2, 1}, {6, 5}, {7, 4}},
                {{4, 5}, {7, 6}, {3, 2}, {0, 1}},
                {{5, 6}, {4, 7}, {0, 3}, {1, 2}},
                {{6, 7}, {5, 4}, {1, 0}, {2, 3}},
                {{7, 4}, {6, 5}, {2, 1}, {3, 0}},
                {{0, 4}, {3, 7}, {2, 6}, {1, 5}},
                {{1, 5}, {0, 4}, {3, 7}, {2, 6}},
                {{2, 6}, {1, 5}, {0, 4}, {3, 7}},
                {{3, 7}, {2, 6}, {1, 5}, {0, 4}},
            };
            const int (*row)[2] = rows[edge];
            t = lv[row[0][0]] / (lv[row[0][0]] - lv[row[0][1]] + eps);
            at = 0.0f;
            bt = lv[row[1][0]] + (lv[row[1][1]] - lv[row[1][0]]) * t;
            ct = lv[row[2][0]] + (lv[row[2][1]] - lv[row[2][0]]) * t;
            dt = lv[row[3][0]] + (lv[row[3][1]] - lv[row[3][0]]) * t;
        }

        int test = 0;
        if (at >= 0.0f)
            test += 1;
        if (bt >= 0.0f)
            test += 2;
        if (ct >= 0.0f)
            test += 4;
        if (dt >= 0.0f)
            test += 8;

        if (test == 5) {
            return (at * ct - bt * dt < eps) ? (s > 0) : false;
        }
        if (test == 10) {
            return (at * ct - bt * dt >= eps) ? (s > 0) : false;
        }
        if (test == 7 || test == 11 || test >= 13) {
            return s < 0;
        }
        return s > 0;
    }

    // Vertex on a Lewiner edge (0-11), or the interior vertex (12): the
    // corner centroid weighted by closeness to the surface.
    __host__ __device__ float3 vertex(int vi) {
        if (vi == 12) {
            float3 sum = make_float3(0.0f, 0.0f, 0.0f);
            float weight = 0.0f;
            for (int i = 0; i < 8; i++) {
                float w = 1.0f / (eps + (lv[i] < 0.0f ? -lv[i] : lv[i]));
                sum = sum + w * lp[i];
                weight += w;
            }
            return sum / weight;
        }

        const int edges[12][2] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
            {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
        };
        int a = edges[vi][0];
        int b = edges[vi][1];
        // Interpolate in a canonical direction (smaller Morton corner
        // first) so shared edges weld across cells.
        if (our_corner(a) > our_corner(b)) {
            int tmp = a;
            a = b;
            b = tmp;
        }
        float denom = lv[b] - lv[a];
        float t = (denom != 0.0f) ? -lv[a] / denom : 0.0f;
        return lerp(t, lp[a], lp[b]);
    }

    // Emit nt triangles whose vertex ids come from get(k).
    template <typename LutFn>
    __host__ __device__ void add_from(LutFn get, int nt) {
        for (int i = 0; i < nt; i++) {
            float3 tri[3];
            for (int j = 0; j < 3; j++) {
                tri[j] = vertex(get(i * 3 + j));
            }
            if (tri[0] != tri[1] && tri[0] != tri[2] && tri[1] != tri[2]) {
                v[v_idx++] = tri[0];
                v[v_idx++] = tri[1];
                v[v_idx++] = tri[2];
            }
        }
    }

    template <typename Table2>
    __host__ __device__ void add(Table2 table, int config, int nt) {
        add_from([&](int k) { return table(luts, config, k); }, nt);
    }

    template <typename Table3>
    __host__ __device__ void add2(Table3 table, int config, int sub, int nt) {
        add_from([&](int k) { return table(luts, config, sub, k); }, nt);
    }

    // The per-class decision procedure: a faithful port of "the big
    // switch" of the reference implementation.
    __host__ __device__ void big_switch(int case_num, int config) {
        int sub = 0;
        switch (case_num) {
        case 1:
            add(mc33::TILING1, config, 1);
            break;
        case 2:
            add(mc33::TILING2, config, 2);
            break;
        case 3:
            if (face_test(mc33::TEST3(luts, config))) {
                add(mc33::TILING3_2, config, 4);
            } else {
                add(mc33::TILING3_1, config, 2);
            }
            break;
        case 4:
            if (interior_test(4, config, 0, mc33::TEST4(luts, config))) {
                add(mc33::TILING4_1, config, 2);
            } else {
                add(mc33::TILING4_2, config, 6);
            }
            break;
        case 5:
            add(mc33::TILING5, config, 3);
            break;
        case 6:
            if (face_test(mc33::TEST6(luts, config, 0))) {
                add(mc33::TILING6_2, config, 5);
            } else if (interior_test(6, config, 0,
                                     mc33::TEST6(luts, config, 1))) {
                add(mc33::TILING6_1_1, config, 3);
            } else {
                add(mc33::TILING6_1_2, config, 9);
            }
            break;
        case 7:
            if (face_test(mc33::TEST7(luts, config, 0)))
                sub += 1;
            if (face_test(mc33::TEST7(luts, config, 1)))
                sub += 2;
            if (face_test(mc33::TEST7(luts, config, 2)))
                sub += 4;
            switch (sub) {
            case 0:
                add(mc33::TILING7_1, config, 3);
                break;
            case 1:
                add2(mc33::TILING7_2, config, 0, 5);
                break;
            case 2:
                add2(mc33::TILING7_2, config, 1, 5);
                break;
            case 3:
                add2(mc33::TILING7_3, config, 0, 9);
                break;
            case 4:
                add2(mc33::TILING7_2, config, 2, 5);
                break;
            case 5:
                add2(mc33::TILING7_3, config, 1, 9);
                break;
            case 6:
                add2(mc33::TILING7_3, config, 2, 9);
                break;
            default:
                if (interior_test(7, config, sub,
                                  mc33::TEST7(luts, config, 3))) {
                    add(mc33::TILING7_4_2, config, 9);
                } else {
                    add(mc33::TILING7_4_1, config, 5);
                }
                break;
            }
            break;
        case 8:
            add(mc33::TILING8, config, 2);
            break;
        case 9:
            add(mc33::TILING9, config, 4);
            break;
        case 10:
            if (face_test(mc33::TEST10(luts, config, 0))) {
                if (face_test(mc33::TEST10(luts, config, 1))) {
                    add(mc33::TILING10_1_1_, config, 4);
                } else {
                    add(mc33::TILING10_2, config, 8);
                }
            } else if (face_test(mc33::TEST10(luts, config, 1))) {
                add(mc33::TILING10_2_, config, 8);
            } else if (interior_test(10, config, 0,
                                     mc33::TEST10(luts, config, 2))) {
                add(mc33::TILING10_1_1, config, 4);
            } else {
                add(mc33::TILING10_1_2, config, 8);
            }
            break;
        case 11:
            add(mc33::TILING11, config, 4);
            break;
        case 12:
            if (face_test(mc33::TEST12(luts, config, 0))) {
                if (face_test(mc33::TEST12(luts, config, 1))) {
                    add(mc33::TILING12_1_1_, config, 4);
                } else {
                    add(mc33::TILING12_2, config, 8);
                }
            } else if (face_test(mc33::TEST12(luts, config, 1))) {
                add(mc33::TILING12_2_, config, 8);
            } else if (interior_test(12, config, 0,
                                     mc33::TEST12(luts, config, 2))) {
                add(mc33::TILING12_1_1, config, 4);
            } else {
                add(mc33::TILING12_1_2, config, 8);
            }
            break;
        case 13:
            if (face_test(mc33::TEST13(luts, config, 0)))
                sub += 1;
            if (face_test(mc33::TEST13(luts, config, 1)))
                sub += 2;
            if (face_test(mc33::TEST13(luts, config, 2)))
                sub += 4;
            if (face_test(mc33::TEST13(luts, config, 3)))
                sub += 8;
            if (face_test(mc33::TEST13(luts, config, 4)))
                sub += 16;
            if (face_test(mc33::TEST13(luts, config, 5)))
                sub += 32;
            sub = mc33::SUBCONFIG13(luts, sub);

            if (sub == 0) {
                add(mc33::TILING13_1, config, 4);
            } else if (sub <= 6) {
                add2(mc33::TILING13_2, config, sub - 1, 6);
            } else if (sub <= 18) {
                add2(mc33::TILING13_3, config, sub - 7, 10);
            } else if (sub <= 22) {
                add2(mc33::TILING13_4, config, sub - 19, 12);
            } else if (sub <= 26) {
                sub -= 23;
                if (interior_test(13, config, sub,
                                  mc33::TEST13(luts, config, 6))) {
                    add2(mc33::TILING13_5_1, config, sub, 6);
                } else {
                    add2(mc33::TILING13_5_2, config, sub, 10);
                }
            } else if (sub <= 38) {
                add2(mc33::TILING13_3_, config, sub - 27, 10);
            } else if (sub <= 44) {
                add2(mc33::TILING13_2_, config, sub - 39, 6);
            } else {
                add(mc33::TILING13_1_, config, 4);
            }
            break;
        case 14:
            add(mc33::TILING14, config, 4);
            break;
        default:
            break;
        }
    }
};

struct process_cube_op {
    float3 *v;
    const uint *cell_indices;
    const GridView view;
    const signed char *luts;
    const float level;

    process_cube_op(float3 *v, const uint *cell_indices, const GridView &view,
                    const signed char *luts, const float level)
        : v(v), cell_indices(cell_indices), view(view), luts(luts),
          level(level) {}

    __host__ __device__ void operator()(uint idx) {
        float3 c_p[8];
        float c_v[8];
        view.load_corners(cell_indices[idx], c_p, c_v);

        cell_worker w;
        w.v = v;
        w.luts = luts;
        w.v_idx = idx * Lewiner::max_triangles * 3;

        int index = 0;
        for (int i = 0; i < 8; i++) {
            w.lv[i] = c_v[our_corner(i)] - level;
            w.lp[i] = c_p[our_corner(i)];
            if (w.lv[i] > 0.0f) {
                index |= 1 << i;
            }
        }

        int case_num = mc33::CASES(luts, index, 0);
        int config = mc33::CASES(luts, index, 1);
        w.big_switch(case_num, config);
    }
};

}   // anonymous namespace

void
Lewiner::run(float3 *v, const uint num_cells, const uint8_t *cases,
             const uint *cell_indices, const GridView &view,
             const float level) {
    // Uploaded once and intentionally leaked so the buffer is not freed
    // after the CUDA context is gone at interpreter shutdown.
    static const thrust::device_vector<signed char> &luts =
        *new thrust::device_vector<signed char>(
            mc33::tables, mc33::tables + mc33::table_size);

    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_cells),
        process_cube_op(v, cell_indices, view, luts.data().get(), level));
}

}   // namespace mc
