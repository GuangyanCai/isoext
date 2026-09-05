#include "mc/vega.cuh"
#include "mc/vega_luts.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

// Marching cubes with topological guarantees and the corrected interior
// test, after Chernyaev's MC33.
//
// This is a port of the MC33_c_library by Vega and Abache (MIT License,
// https://github.com/dvega68/MC33_c_library), the reference implementation
// of Vega, Abache and Coll, "A Fast and Memory-Saving Marching Cubes 33
// Implementation with the Correct Interior Test" (Journal of Computer
// Graphics Techniques, 2019). It fixes the problems of the Lewiner
// implementation identified by Custodio et al., "Practical considerations
// on Marching Cubes 33 topological correctness" (Computers & Graphics,
// 2013): the interior test analyzes the trilinear interpolant on the
// plane through its saddle points instead of a single fixed section, so
// tunnels are detected reliably and the result is invariant under
// reflections of the input field.
//
// Cells are processed in Vega's corner numbering and sign convention
// (corner values are level - value, a set case bit means the field is
// above the level) and converted to ours only when vertices are emitted.
// The case dispatch computes an offset into the lookup table, where each
// 16-bit entry packs one triangle: three vertex nibbles (edges 0-11 or 12
// for the cube center) plus a continuation nibble.

namespace mc {

namespace {
static MCRegistrar<Vega> registrar("vega");

static_assert(Vega::max_triangles == (size_t) vega::max_triangles,
              "buffer sizing must match the generated table");

// Vega corner i sits at our Morton corner our_corner[i].
__host__ __device__ inline int
our_corner(int vega) {
    const int ours[8] = {0, 2, 3, 1, 4, 6, 7, 5};
    return ours[vega];
}

// Per-thread state and logic for one cell.
struct cell_worker {
    float3 *out;
    const unsigned short *luts;
    uint v_idx;
    float v[8];    // level minus corner value, Vega order
    float3 p[8];   // corner positions, Vega order

    // Sign of the bilinear interpolant at the saddle point of each face
    // whose four corner signs alternate: 1 joins the positive vertices,
    // -1 the negative ones, 0 means the face is not ambiguous. Returns
    // the sum over all six faces.
    __host__ __device__ int face_tests(int *face, int ind) {
        if (ind & 0x80) {   // vertex 0
            face[0] =
                ((ind & 0xCC) == 0x84 ? (v[0] * v[5] < v[1] * v[4] ? -1 : 1)
                                      : 0);
            face[3] =
                ((ind & 0x99) == 0x81 ? (v[0] * v[7] < v[3] * v[4] ? -1 : 1)
                                      : 0);
            face[4] =
                ((ind & 0xF0) == 0xA0 ? (v[0] * v[2] < v[1] * v[3] ? -1 : 1)
                                      : 0);
        } else {
            face[0] =
                ((ind & 0xCC) == 0x48 ? (v[0] * v[5] < v[1] * v[4] ? 1 : -1)
                                      : 0);
            face[3] =
                ((ind & 0x99) == 0x18 ? (v[0] * v[7] < v[3] * v[4] ? 1 : -1)
                                      : 0);
            face[4] =
                ((ind & 0xF0) == 0x50 ? (v[0] * v[2] < v[1] * v[3] ? 1 : -1)
                                      : 0);
        }
        if (ind & 0x02) {   // vertex 6
            face[1] =
                ((ind & 0x66) == 0x42 ? (v[1] * v[6] < v[2] * v[5] ? -1 : 1)
                                      : 0);
            face[2] =
                ((ind & 0x33) == 0x12 ? (v[3] * v[6] < v[2] * v[7] ? -1 : 1)
                                      : 0);
            face[5] =
                ((ind & 0x0F) == 0x0A ? (v[4] * v[6] < v[5] * v[7] ? -1 : 1)
                                      : 0);
        } else {
            face[1] =
                ((ind & 0x66) == 0x24 ? (v[1] * v[6] < v[2] * v[5] ? 1 : -1)
                                      : 0);
            face[2] =
                ((ind & 0x33) == 0x21 ? (v[3] * v[6] < v[2] * v[7] ? 1 : -1)
                                      : 0);
            face[5] =
                ((ind & 0x0F) == 0x05 ? (v[4] * v[6] < v[5] * v[7] ? 1 : -1)
                                      : 0);
        }
        return face[0] + face[1] + face[2] + face[3] + face[4] + face[5];
    }

    // Same test for a single face; returns the mask of the joined
    // vertices. Only used for cases 3 and 6.
    __host__ __device__ int face_test1(int face) {
        switch (face) {
        case 0:
            return v[0] * v[5] < v[1] * v[4] ? 0x48 : 0x84;
        case 1:
            return v[1] * v[6] < v[2] * v[5] ? 0x24 : 0x42;
        case 2:
            return v[3] * v[6] < v[2] * v[7] ? 0x21 : 0x12;
        case 3:
            return v[0] * v[7] < v[3] * v[4] ? 0x18 : 0x81;
        case 4:
            return v[0] * v[2] < v[1] * v[3] ? 0x50 : 0xA0;
        default:
            return v[4] * v[6] < v[5] * v[7] ? 0x05 : 0x0A;
        }
    }

    // The corrected interior test: nonzero if the diagonally opposite
    // vertices i and i+6 mod 8 are joined through the cell interior. The
    // trilinear interpolant is analyzed on the section through its saddle
    // points, not on a fixed one. With flag13 set (case 13 only), returns
    // 2 if one of the vertices 0-3 joins the cube center and 1 if one of
    // the vertices 4-7 does, distinguishing the 13.5.2 orientations.
    __host__ __device__ int interior_test(int i, int flag13) {
        float at = v[4] - v[0], bt = v[5] - v[1];
        float ct = v[6] - v[2], dt = v[7] - v[3];
        float t = at * ct - bt * dt;   // the "a" value
        if (signbit(t)) {
            if (i & 0x01)
                return 0;
        } else {
            if (!(i & 0x01) || t == 0.0f)
                return 0;
        }
        t = 0.5f * (v[3] * bt - v[2] * at + v[1] * dt - v[0] * ct) /
            t;   // -b/2a
        if (t > 0.0f && t < 1.0f) {
            at = v[0] + at * t;
            bt = v[1] + bt * t;
            ct = v[2] + ct * t;
            dt = v[3] + dt * t;
            ct *= at;
            dt *= bt;
            if (i & 0x01) {
                if (ct < dt && !signbit(dt))
                    return (signbit(bt) == signbit(v[i])) + flag13;
            } else {
                if (ct > dt && !signbit(ct))
                    return (signbit(at) == signbit(v[i])) + flag13;
            }
        }
        return 0;
    }

    // Vertex on a Vega edge (0-11), or the interior vertex (12) at the
    // cube center.
    __host__ __device__ float3 vertex(int vi) {
        if (vi == 12) {
            return 0.5f * (p[0] + p[6]);
        }

        const int edges[12][2] = {
            {0, 1}, {1, 2}, {3, 2}, {0, 3}, {4, 5}, {5, 6},
            {7, 6}, {4, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
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
        float denom = v[a] - v[b];
        float t = (denom != 0.0f) ? v[a] / denom : 0.0f;
        return lerp(t, p[a], p[b]);
    }

    // Emit the triangle strip starting after the given table offset. Each
    // entry is one triangle; the top nibble marks continuation. With
    // reversed set the first two vertices are swapped.
    __host__ __device__ void emit(int offset, bool reversed) {
        int idx = offset;
        unsigned int e;
        do {
            e = luts[++idx];
            float3 tri[3];
            tri[2] = vertex(e & 0xF);
            tri[1] = vertex((e >> 4) & 0xF);
            tri[0] = vertex((e >> 8) & 0xF);
            if (tri[0] != tri[1] && tri[0] != tri[2] && tri[1] != tri[2]) {
                out[v_idx++] = tri[reversed ? 1 : 0];
                out[v_idx++] = tri[reversed ? 0 : 1];
                out[v_idx++] = tri[2];
            }
        } while (e >> 12);
    }

    // The case dispatch: a faithful port of MC33_findCase of the
    // reference implementation. Bit 7 of the case index i is vertex 0,
    // bit 0 is vertex 7; indices with vertex 0 set use the table entry of
    // the complement with reversed orientation.
    __host__ __device__ void find_case(unsigned int i) {
        int f[6];
        unsigned int c, m, n;
        if (i & 0x80) {
            c = luts[i ^ 0xFF];
            m = (c & 0x800) == 0;
            n = !m;
        } else {
            c = luts[i];
            n = (c & 0x800) == 0;
            m = !n;
        }
        int k = c & 0x7FF;
        int offset = 0;
        switch (c >> 12) {   // find the MC33 case
        case 0:              // cases 1, 2, 5, 8, 9, 11 and 14
            offset = k;
            break;
        case 1:   // case 3
            offset = ((m ? i : i ^ 0xFF) & face_test1(k >> 2)) ? 183 + 2 * k
                                                               : 159 + k;
            break;
        case 2:   // case 4
            offset = interior_test(k, 0) ? 239 + 6 * k : 231 + 2 * k;
            break;
        case 3:   // case 6
            if ((m ? i : i ^ 0xFF) & face_test1(k % 6)) {
                offset = 575 + 5 * k;   // 6.2
            } else {
                offset = interior_test(k / 6, 0) ? 407 + 7 * k
                                                 : 335 + 3 * k;   // 6.1
            }
            break;
        case 4:   // case 7
            switch (face_tests(f, m ? i : i ^ 0xFF)) {
            case -3:
                offset = 695 + 3 * k;   // 7.1
                break;
            case -1:   // 7.2
                offset =
                    (f[4] + f[5] < 0 ? (f[0] + f[2] < 0 ? 759 : 799) : 719) +
                    5 * k;
                break;
            case 1:   // 7.3
                offset =
                    (f[4] + f[5] < 0 ? 983 : (f[0] + f[2] < 0 ? 839 : 911)) +
                    9 * k;
                break;
            default:   // 7.4
                offset = interior_test(k >> 1, 0) ? 1095 + 9 * k : 1055 + 5 * k;
            }
            break;
        case 5:   // case 10
            switch (face_tests(f, m ? i : i ^ 0xFF)) {
            case -2:
                if (k == 2
                        ? interior_test(0, 0)
                        : interior_test(0, 0) || interior_test(k ? 1 : 3, 0)) {
                    offset = 1213 + 8 * k;   // 10.1.2
                } else {
                    offset = 1189 + 4 * k;   // 10.1.1
                }
                break;
            case 0:   // 10.2
                offset = (f[2 + k] < 0 ? 1261 : 1285) + 8 * k;
                break;
            default:
                if (k == 2
                        ? interior_test(1, 0)
                        : interior_test(2, 0) || interior_test(k ? 3 : 1, 0)) {
                    offset = 1237 + 8 * k;   // 10.1.2
                } else {
                    offset = 1201 + 4 * k;   // 10.1.1
                }
            }
            break;
        case 6:   // case 12
            switch (face_tests(f, m ? i : i ^ 0xFF)) {
            case -2:   // 12.1
                offset = interior_test((0xDA010C >> (k << 1)) & 3, 0)
                             ? 1453 + 8 * k
                             : 1357 + 4 * k;
                break;
            case 0:   // 12.2
                offset = (f[k >> 1] < 0 ? 1645 : 1741) + 8 * k;
                break;
            default:   // 12.1
                offset = interior_test((0xA7B7E5 >> (k << 1)) & 3, 0)
                             ? 1549 + 8 * k
                             : 1405 + 4 * k;
            }
            break;
        default:   // case 13
            switch (abs(face_tests(f, 165))) {
            case 0: {
                k = ((f[1] < 0) << 1) | (f[5] < 0);
                if (f[0] * f[1] == f[5]) {   // 13.4
                    offset = 2157 + 12 * k;
                } else {   // 13.5.1 if joined == 0 else 13.5.2
                    int joined = interior_test(k, 1);
                    offset = 2285 + (joined ? 10 * k - 40 * joined : 6 * k);
                }
                break;
            }
            case 2:   // 13.3
                offset =
                    1917 + 10 * ((f[0] < 0 ? (f[2] > 0) : 12 + (f[2] < 0)) +
                                 (f[1] < 0 ? (f[3] < 0) : 6 + (f[3] > 0)));
                if (f[4] > 0)
                    offset += 30;
                break;
            case 4:   // 13.2
                k = 21 + 11 * f[0] + 4 * f[1] + 3 * f[2] + 2 * f[3] + f[4];
                if (k >> 4)
                    k -= (k & 32) ? 20 : 10;
                offset = 1845 + 3 * k;
                break;
            default:   // 13.1
                offset = 1839 + 2 * f[0];
            }
        }
        // m rather than n: the reference implementation's winding is
        // mirrored relative to ours.
        emit(offset, m != 0);
    }
};

struct process_cube_op {
    float3 *out;
    const uint *cell_indices;
    const GridView view;
    const unsigned short *luts;
    const float level;

    process_cube_op(float3 *out, const uint *cell_indices, const GridView &view,
                    const unsigned short *luts, const float level)
        : out(out), cell_indices(cell_indices), view(view), luts(luts),
          level(level) {}

    __host__ __device__ void operator()(uint idx) {
        float3 c_p[8];
        float c_v[8];
        view.load_corners(cell_indices[idx], c_p, c_v);

        cell_worker w;
        w.out = out;
        w.luts = luts;
        w.v_idx = idx * Vega::max_triangles * 3;

        unsigned int index = 0;
        for (int i = 0; i < 8; i++) {
            w.v[i] = level - c_v[our_corner(i)];
            w.p[i] = c_p[our_corner(i)];
            if (signbit(w.v[i])) {
                index |= 0x80 >> i;
            }
        }

        if (index && index != 0xFF) {
            w.find_case(index);
        }
    }
};

}   // anonymous namespace

void
Vega::run(float3 *v, const uint num_cells, const uint8_t *cases,
          const uint *cell_indices, const GridView &view, const float level) {
    // Uploaded once and intentionally leaked so the buffer is not freed
    // after the CUDA context is gone at interpreter shutdown.
    static const thrust::device_vector<unsigned short> &luts =
        *new thrust::device_vector<unsigned short>(
            vega::tables, vega::tables + vega::table_size);

    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_cells),
        process_cube_op(v, cell_indices, view, luts.data().get(), level));
}

}   // namespace mc
