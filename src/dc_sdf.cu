#include "dc.cuh"
#include "math.cuh"
#include "shared_luts.cuh"
#include "sym3x3.cuh"
#include "utils.cuh"

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

// Dual contouring of signed distance data, after Carrera, Wang, Batty,
// Stein and Sellán (SIGGRAPH 2026). Connectivity is that of dual
// contouring; only the vertex placement differs. Every grid sample is a
// sphere of radius |s| that the surface must touch, and each cell's vertex
// is optimized so a local mesh around it (fans from the vertex over the
// cell's Hermite points and the points where mesh edges cross its faces)
// is tangent to the spheres assigned to the cell. Hermite data is
// estimated from the trilinear interpolant and refined from the mesh
// itself, so no normals are needed. Defaults and formulas follow the
// authors' reference code; where it differs from the paper (the Hermite
// normal blend, radius-weighted sphere rows, and the mesh used for the
// first sample assignment) the code was measurably better on the reference
// runs, and the comments below mark the spots.
//
// One deviation from the reference: samples farther than `band` cell
// diagonals from the surface are left out instead of being drawn into a
// random batch. In our comparison against the reference this lost nothing
// (it was slightly more accurate) and keeps the closest-point search local
// and the result deterministic.

namespace {

// Face directions: 0:-x 1:+x 2:-y 3:+y 4:-z 5:+z, and the four cell edges
// (shared_luts numbering) bounding each face.
__host__ __device__ inline int3
face_dir(int d) {
    int s = (d & 1) ? 1 : -1;
    return make_int3(d < 2 ? s : 0, (d >= 2 && d < 4) ? s : 0, d >= 4 ? s : 0);
}

__host__ __device__ inline int
face_edge(int d, int k) {
    const int table[6][4] = {{0, 1, 2, 3},   {4, 5, 6, 7},  {0, 9, 4, 8},
                             {2, 10, 6, 11}, {3, 11, 7, 8}, {1, 10, 5, 9}};
    return table[d][k];
}

__host__ __device__ inline float3
to_float3(uint3 v) {
    return make_float3(v.x, v.y, v.z);
}

__host__ __device__ inline float3
normalize_or(float3 v, float3 fallback) {
    float len = norm(v);
    return len > 0.0f ? v / len : fallback;
}

// Index of the first element of a sorted array that is not less than key
// (n when there is none). Written out because thrust::lower_bound with
// thrust::seq gave wrong results inside the ring search kernel.
template <typename T, typename Less>
__host__ __device__ inline uint
lower_bound_index(const T *arr, uint n, T key, Less less) {
    uint lo = 0, hi = n;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (less(arr[mid], key)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

struct uint_less {
    __host__ __device__ bool operator()(uint a, uint b) const { return a < b; }
};

// Maps dense cell coordinates to positions in its.cell_indices (the dual
// vertex index), or -1 when the cell is outside the grid or not crossed.
struct CellLookup {
    GridView view;
    uint num_grid_cells;
    const uint *its_cells;
    uint num_its_cells;

    __host__ __device__ int find(const uint *arr, uint n, uint key) const {
        uint i = lower_bound_index(arr, n, key, uint_less());
        return (i < n && arr[i] == key) ? int(i) : -1;
    }

    __host__ __device__ int operator()(int3 c) const {
        uint3 ncells = view.shape - 1;
        if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= int(ncells.x) ||
            c.y >= int(ncells.y) || c.z >= int(ncells.z)) {
            return -1;
        }
        uint cell = idx_3d_to_1d(make_uint3(c.x, c.y, c.z), ncells);
        if (view.sparse) {
            int pos = find(view.cell_indices, num_grid_cells, cell);
            if (pos < 0) {
                return -1;
            }
            cell = pos;
        }
        return find(its_cells, num_its_cells, cell);
    }

    // Dense coordinates of the cell behind a dual vertex index.
    __host__ __device__ int3 coords(uint dual) const {
        uint cell = its_cells[dual];
        uint dense = view.sparse ? view.cell_indices[cell] : cell;
        uint3 c = idx_1d_to_3d(dense, view.shape - 1);
        return make_int3(c.x, c.y, c.z);
    }

    __host__ __device__ int3 coords_of_point(float3 p) const {
        uint3 ncells = view.shape - 1;
        float3 t = (p - view.aabb_min) / (view.aabb_max - view.aabb_min);
        int3 c = make_int3(floorf(t.x * ncells.x), floorf(t.y * ncells.y),
                           floorf(t.z * ncells.z));
        return make_int3(min(max(c.x, 0), int(ncells.x) - 1),
                         min(max(c.y, 0), int(ncells.y) - 1),
                         min(max(c.z, 0), int(ncells.z) - 1));
    }
};

// One thread per crossed cell: find each crossing's local edge and unique
// edge index, seed the Hermite data (Eq. 1 and 2: the crossing, and the
// trilinear gradient summed over the cells sharing the edge) and start the
// vertex at the centroid of the cell's crossings (Eq. 3).
struct prep_cells_op {
    uint8_t *local_edge;   // per crossing
    uint *edge_of;         // per crossing: unique edge index
    float3 *g;             // per crossing: unit gradient
    float3 *h;             // per unique edge
    float3 *edge_a, *edge_b;
    float3 *x;   // per cell
    const float3 *its_points;
    const float3 *its_normals;   // null: estimate from the grid
    const uint2 *its_edges;
    const uint *its_offsets;
    const uint *its_cells;
    const uint2 *unique_edges;
    const uint num_unique;
    const GridView view;
    const int *edges_tab;

    __host__ __device__ int corner_of(uint cell, uint point_id) const {
        if (view.sparse) {
            return point_id % 8;
        }
        for (uint c = 0; c < 8; c++) {
            if (view.corner_point_id(cell, c) == point_id) {
                return c;
            }
        }
        return 0;
    }

    __host__ __device__ void operator()(uint i) {
        uint cell = its_cells[i];
        uint begin = its_offsets[i], end = its_offsets[i + 1];
        float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
        for (uint m = begin; m < end; m++) {
            int c0 = corner_of(cell, its_edges[m].x);
            int c1 = corner_of(cell, its_edges[m].y);
            for (int e = 0; e < 12; e++) {
                if (edges_tab[2 * e] == c0 && edges_tab[2 * e + 1] == c1) {
                    local_edge[m] = e;
                }
            }
            uint2 key = make_uint2(view.dense_point_index(cell, c0),
                                   view.dense_point_index(cell, c1));
            uint e = lower_bound_index(unique_edges, num_unique, key,
                                       uint2_less_pred());
            edge_of[m] = e;

            float3 p = its_points[m];
            // Like the reference, unit gradients are averaged over the
            // cells sharing the edge (done after this kernel).
            g[m] = normalize_or(its_normals ? its_normals[m]
                                            : view.gradient_in_cell(cell, p),
                                make_float3(0.0f, 0.0f, 0.0f));
            h[e] = p;
            edge_a[e] = view.corner_position(cell, c0);
            edge_b[e] = view.corner_position(cell, c1);
            centroid = centroid + p;
        }
        x[i] = centroid / float(end - begin);
    }
};

// One thread per cell: the dual contouring QEF of the estimated Hermite
// data, degenerate directions falling back to the centroid, unclamped.
// The reference assigns the samples of its first outer iteration on this
// mesh (its vertices are reset to the centroids only afterwards), and that
// turned out clearly more accurate than assigning on the centroid mesh
// the paper describes, so it is kept as an option.
struct qef_vertices_op {
    float3 *x_qef;
    const float3 *x;   // centroids
    const float3 *h;
    const float3 *n;
    const uint *edge_of;
    const uint *its_offsets;

    __host__ __device__ void operator()(uint i) {
        float ATA[3][3] = {};
        float3 ATb = make_float3(0.0f, 0.0f, 0.0f);
        for (uint m = its_offsets[i]; m < its_offsets[i + 1]; m++) {
            float3 a = n[edge_of[m]];
            ATA[0][0] += a.x * a.x;
            ATA[0][1] += a.x * a.y;
            ATA[0][2] += a.x * a.z;
            ATA[1][1] += a.y * a.y;
            ATA[1][2] += a.y * a.z;
            ATA[2][2] += a.z * a.z;
            ATb = ATb + a * dot(a, h[edge_of[m]]);
        }
        ATA[1][0] = ATA[0][1];
        ATA[2][0] = ATA[0][2];
        ATA[2][1] = ATA[1][2];
        float3 c = x[i];
        float3 rhs =
            ATb -
            make_float3(dot(make_float3(ATA[0][0], ATA[0][1], ATA[0][2]), c),
                        dot(make_float3(ATA[1][0], ATA[1][1], ATA[1][2]), c),
                        dot(make_float3(ATA[2][0], ATA[2][1], ATA[2][2]), c));
        x_qef[i] = c + solve_sym_3x3(ATA, rhs, 1e-2f);
    }
};

struct in_band_pred {
    float level, max_radius;
    __host__ __device__ bool operator()(float v) const {
        return fabsf(v - level) < max_radius;
    }
};

// Grid samples inside the band: position, radius |s - level| and sign.
struct Samples {
    thrust::device_vector<float3> u;
    thrust::device_vector<float> radius;
    thrust::device_vector<bool> inside;
};

Samples
gather_samples(const GridView &view, uint num_grid_cells, float level,
               float max_radius) {
    // Dense lattice ids of the candidate samples; a sparse grid stores its
    // values per cell corner, so shared corners are deduplicated first.
    thrust::device_vector<uint> ids;
    thrust::device_vector<float> values;
    if (view.sparse) {
        uint num_corners = num_grid_cells * 8;
        ids.resize(num_corners);
        values.resize(num_corners);
        thrust::for_each(thrust::counting_iterator<uint>(0),
                         thrust::counting_iterator<uint>(num_corners),
                         [ids = ids.data().get(), values = values.data().get(),
                          view] __device__(uint cid) {
                             ids[cid] =
                                 view.dense_point_index(cid / 8, cid % 8);
                             values[cid] = view.values[cid];
                         });
        thrust::sort_by_key(ids.begin(), ids.end(), values.begin());
        auto end =
            thrust::unique_by_key(ids.begin(), ids.end(), values.begin());
        ids.erase(end.first, ids.end());
        values.erase(end.second, values.end());
    } else {
        uint num_points = view.shape.x * view.shape.y * view.shape.z;
        ids.resize(num_points);
        thrust::sequence(ids.begin(), ids.end());
        thrust::device_ptr<const float> v(view.values);
        values.assign(v, v + num_points);
    }

    in_band_pred in_band{level, max_radius};
    thrust::device_vector<uint> kept(ids.size());
    auto kept_end = thrust::copy_if(ids.begin(), ids.end(), values.begin(),
                                    kept.begin(), in_band);
    kept.erase(kept_end, kept.end());
    thrust::device_vector<float> kept_values(values.size());
    auto values_end = thrust::copy_if(values.begin(), values.end(),
                                      kept_values.begin(), in_band);
    kept_values.erase(values_end, kept_values.end());
    values = std::move(kept_values);

    Samples s;
    uint num = kept.size();
    s.u.resize(num);
    s.radius.resize(num);
    s.inside.resize(num);
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num),
                     [ids = kept.data().get(), values = values.data().get(),
                      u = s.u.data().get(), radius = s.radius.data().get(),
                      inside = s.inside.data().get(), view,
                      level] __device__(uint j) {
                         u[j] = get_vtx_pos_op(view.shape, view.aabb_min,
                                               view.aabb_max)(ids[j]);
                         float v = values[j] - level;
                         radius[j] = fabsf(v);
                         inside[j] = v < 0.0f;
                     });
    return s;
}

// Triangles of the current global mesh: two per crossed edge whose four
// cells all carry a vertex, split along the quad's first diagonal like the
// reference. Vertices are dual indices into x.
struct TriangleSoup {
    thrust::device_vector<uint3> tris;      // dual indices
    thrust::device_vector<uint> bin_keys;   // dense cell id, sorted
    thrust::device_vector<uint> bin_tris;   // triangle per bin entry
    thrust::device_vector<uint> occupied;   // one bit per dense cell

    __host__ __device__ static float3 vertex(const uint3 *tris, const float3 *x,
                                             uint t, int k) {
        uint3 tri = tris[t];
        return x[k == 0 ? tri.x : (k == 1 ? tri.y : tri.z)];
    }
};

// Cell range covered by the bounding box of a triangle.
struct tri_cell_range_op {
    const uint3 *tris;
    const float3 *x;
    const CellLookup lookup;

    __host__ __device__ void operator()(uint t, int3 &lo, int3 &hi) const {
        float3 a = TriangleSoup::vertex(tris, x, t, 0);
        float3 b = TriangleSoup::vertex(tris, x, t, 1);
        float3 c = TriangleSoup::vertex(tris, x, t, 2);
        float3 bmin = make_float3(fminf(a.x, fminf(b.x, c.x)),
                                  fminf(a.y, fminf(b.y, c.y)),
                                  fminf(a.z, fminf(b.z, c.z)));
        float3 bmax = make_float3(fmaxf(a.x, fmaxf(b.x, c.x)),
                                  fmaxf(a.y, fmaxf(b.y, c.y)),
                                  fmaxf(a.z, fmaxf(b.z, c.z)));
        lo = lookup.coords_of_point(bmin);
        hi = lookup.coords_of_point(bmax);
    }
};

// Bin the triangles into every grid cell their bounding box overlaps so a
// ring search over cells finds every triangle within a given distance. The
// occupancy bits let the search skip empty cells without a binary search.
void
bin_triangles(TriangleSoup &soup, const thrust::device_vector<float3> &x,
              const CellLookup &lookup) {
    uint num_tris = soup.tris.size();
    tri_cell_range_op range{soup.tris.data().get(), x.data().get(), lookup};
    uint3 ncells = lookup.view.shape - 1;
    soup.occupied.assign((ncells.x * ncells.y * ncells.z + 31) / 32, 0);

    thrust::device_vector<uint> offsets(num_tris + 1, 0);
    thrust::transform(thrust::counting_iterator<uint>(0),
                      thrust::counting_iterator<uint>(num_tris),
                      offsets.begin(), [range] __device__(uint t) {
                          int3 lo, hi;
                          range(t, lo, hi);
                          return uint((hi.x - lo.x + 1) * (hi.y - lo.y + 1) *
                                      (hi.z - lo.z + 1));
                      });
    thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
    uint num_entries = offsets[num_tris];

    soup.bin_keys.resize(num_entries);
    soup.bin_tris.resize(num_entries);
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_tris),
        [range, offsets = offsets.data().get(),
         keys = soup.bin_keys.data().get(), tris = soup.bin_tris.data().get(),
         occupied = soup.occupied.data().get(), ncells] __device__(uint t) {
            int3 lo, hi;
            range(t, lo, hi);
            uint k = offsets[t];
            for (int z = lo.z; z <= hi.z; z++) {
                for (int y = lo.y; y <= hi.y; y++) {
                    for (int x = lo.x; x <= hi.x; x++) {
                        uint key = idx_3d_to_1d(make_uint3(x, y, z), ncells);
                        keys[k] = key;
                        tris[k] = t;
                        atomicOr(&occupied[key / 32], 1u << (key % 32));
                        k++;
                    }
                }
            }
        });
    thrust::sort_by_key(soup.bin_keys.begin(), soup.bin_keys.end(),
                        soup.bin_tris.begin());
}

// Warp-wide reductions.
__device__ inline float
warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    return v;
}

__device__ inline float3
warp_sum(float3 v) {
    return make_float3(warp_sum(v.x), warp_sum(v.y), warp_sum(v.z));
}

// Lane holding the smallest value; ties go to the lowest lane.
__device__ inline int
warp_argmin(float v) {
    int lane = threadIdx.x & 31;
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, v, o);
        int ol = __shfl_xor_sync(0xffffffff, lane, o);
        if (ov < v || (ov == v && ol < lane)) {
            v = ov;
            lane = ol;
        }
    }
    return lane;
}

// One warp per sample: closest point on the global mesh within
// radius + one cell diagonal (the reference's outlier cutoff), searched
// over rings of cells around the sample with the lanes sharing each ring,
// then the cell containing that point. Writes the dual index of that cell,
// or -1.
__global__ void
assign_samples_kernel(int *cell_of, const float3 *u, const float *radius,
                      uint num_samples, const uint3 *tris, const float3 *x,
                      const uint *bin_keys, const uint *bin_tris, uint num_bins,
                      const uint *occupied, CellLookup lookup, float diag,
                      float h_min) {
    uint j = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (j >= num_samples) {
        return;
    }
    float3 p = u[j];
    float max_dist = radius[j] + diag;
    float best = max_dist * max_dist;
    float3 best_pt = p;

    uint3 ncells = lookup.view.shape - 1;
    float3 cell_size =
        (lookup.view.aabb_max - lookup.view.aabb_min) / to_float3(ncells);
    int3 c0 = lookup.coords_of_point(p);
    // Cells `ring` steps away are at least (ring - 1) cells from p.
    for (int ring = 0; (ring - 1) * h_min < max_dist; ring++) {
        int side = 2 * ring + 1;
        for (int idx = lane; idx < side * side * side; idx += 32) {
            int dx = idx % side - ring;
            int dy = (idx / side) % side - ring;
            int dz = idx / (side * side) - ring;
            if (max(abs(dx), max(abs(dy), abs(dz))) != ring) {
                continue;
            }
            int3 c = make_int3(c0.x + dx, c0.y + dy, c0.z + dz);
            if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= int(ncells.x) ||
                c.y >= int(ncells.y) || c.z >= int(ncells.z)) {
                continue;
            }
            uint key = idx_3d_to_1d(make_uint3(c.x, c.y, c.z), ncells);
            if (!(occupied[key / 32] >> (key % 32) & 1u)) {
                continue;
            }
            // Skip cells whose box is already farther than the best hit.
            float3 lo_pt =
                lookup.view.aabb_min + make_float3(c.x, c.y, c.z) * cell_size;
            float3 gap = make_float3(
                fmaxf(fmaxf(lo_pt.x - p.x, p.x - lo_pt.x - cell_size.x), 0.0f),
                fmaxf(fmaxf(lo_pt.y - p.y, p.y - lo_pt.y - cell_size.y), 0.0f),
                fmaxf(fmaxf(lo_pt.z - p.z, p.z - lo_pt.z - cell_size.z), 0.0f));
            if (dot(gap, gap) >= best) {
                continue;
            }
            uint lo = lower_bound_index(bin_keys, num_bins, key, uint_less());
            for (uint it = lo; it < num_bins && bin_keys[it] == key; it++) {
                uint t = bin_tris[it];
                float3 q = closest_point_triangle(
                    p, TriangleSoup::vertex(tris, x, t, 0),
                    TriangleSoup::vertex(tris, x, t, 1),
                    TriangleSoup::vertex(tris, x, t, 2));
                float d2 = dot(q - p, q - p);
                if (d2 < best) {
                    best = d2;
                    best_pt = q;
                }
            }
        }
        int src = warp_argmin(best);
        best = __shfl_sync(0xffffffff, best, src);
        best_pt.x = __shfl_sync(0xffffffff, best_pt.x, src);
        best_pt.y = __shfl_sync(0xffffffff, best_pt.y, src);
        best_pt.z = __shfl_sync(0xffffffff, best_pt.z, src);
        if (best < max_dist * max_dist &&
            best <= (ring * h_min) * (ring * h_min)) {
            break;
        }
    }
    if (lane == 0) {
        cell_of[j] = best < max_dist * max_dist
                         ? lookup(lookup.coords_of_point(best_pt))
                         : -1;
    }
}

// One thread per cell: where the mesh edges to the six face neighbors
// cross the cell's faces, blended with the previous points.
struct face_points_op {
    float3 *p;   // 6 per cell, NAN when absent
    const float3 *x;
    const CellLookup lookup;
    const float w;

    __host__ __device__ void operator()(uint i) {
        int3 c = lookup.coords(i);
        uint cell = lookup.its_cells[i];
        float3 lo = lookup.view.corner_position(cell, 0);
        float3 hi = lookup.view.corner_position(cell, 7);
        float3 xi = x[i];
        for (int d = 0; d < 6; d++) {
            int3 dir = face_dir(d);
            int nb = lookup(make_int3(c.x + dir.x, c.y + dir.y, c.z + dir.z));
            if (nb < 0) {
                continue;
            }
            float3 seg = x[nb] - xi;
            int axis = d / 2;
            float plane = (d & 1) ? (axis == 0   ? hi.x
                                     : axis == 1 ? hi.y
                                                 : hi.z)
                                  : (axis == 0   ? lo.x
                                     : axis == 1 ? lo.y
                                                 : lo.z);
            float start = axis == 0 ? xi.x : axis == 1 ? xi.y : xi.z;
            float step = axis == 0 ? seg.x : axis == 1 ? seg.y : seg.z;
            if (fabsf(step) < 1e-9f) {
                continue;
            }
            float t = (plane - start) / step;
            if (t < -1e-6f || t > 1.0f + 1e-6f) {
                continue;
            }
            float3 pt = xi + seg * t;
            float3 &slot = p[i * 6 + d];
            slot = isnan(slot.x) ? pt : lerp(w, slot, pt);
        }
    }
};

// One thread per unique edge: refit the Hermite data from the best-fit
// plane of the four vertices around the edge (Eq. 7). The normal blend is
// the reference code's (1 - w) old + w new, not the paper's (new + w old);
// the code's was more accurate in our runs.
struct hermite_update_op {
    float3 *h;
    float3 *n;
    const float3 *x;
    const int4 *quads;   // dual indices, -1 when missing
    const float3 *edge_a, *edge_b;
    const bool *a_inside;
    const float w;

    __host__ __device__ void operator()(uint e) {
        int4 q = quads[e];
        if (q.x < 0 || q.y < 0 || q.z < 0 || q.w < 0) {
            return;
        }
        float3 v[4] = {x[q.x], x[q.y], x[q.z], x[q.w]};
        float3 centroid = (v[0] + v[1] + v[2] + v[3]) / 4.0f;
        float cov[3][3] = {};
        for (int k = 0; k < 4; k++) {
            float3 d = v[k] - centroid;
            cov[0][0] += d.x * d.x;
            cov[0][1] += d.x * d.y;
            cov[0][2] += d.x * d.z;
            cov[1][1] += d.y * d.y;
            cov[1][2] += d.y * d.z;
            cov[2][2] += d.z * d.z;
        }
        cov[1][0] = cov[0][1];
        cov[2][0] = cov[0][2];
        cov[2][1] = cov[1][2];
        float eig[3], V[3][3];
        sym_eigen_3x3(cov, eig, V);
        int k = eig[0] <= eig[1] ? (eig[0] <= eig[2] ? 0 : 2)
                                 : (eig[1] <= eig[2] ? 1 : 2);
        float3 normal =
            normalize_or(make_float3(V[0][k], V[1][k], V[2][k]), n[e]);

        // Orient toward increasing values, like the estimated normals.
        float3 a = edge_a[e], b = edge_b[e];
        float3 dir = normalize_or(b - a, make_float3(0.0f, 0.0f, 1.0f));
        float3 outward = dir * (a_inside[e] ? -1.0f : 1.0f);
        if (dot(normal, outward) > 0.0f) {
            normal = normal * -1.0f;
        }
        n[e] = normalize_or(lerp(w, n[e], normal), n[e]);

        float denom = dot(normal, dir);
        if (fabsf(denom) > 1e-6f) {
            float t = dot(normal, centroid - a) / denom;
            if (t >= 0.0f && t <= norm(b - a)) {
                h[e] = lerp(w, h[e], a + dir * t);
            }
        }
    }
};

// One warp per cell: the inner loop (Eq. 11). Each iteration rebuilds the
// local fan mesh at the current vertex, takes the closest point of every
// assigned sphere on it (lanes share the spheres), and solves the
// linearized least squares for the next vertex until the step drops below
// tol. Every lane accumulates its rows, the warp sums them, and every lane
// solves the same 3x3 system so the vertex stays warp-uniform.
__global__ void
inner_loop_kernel(float3 *x, const float3 *p, const float3 *h, const float3 *n,
                  const uint *edge_of, const uint8_t *local_edge,
                  const uint *its_offsets, uint num_cells, const float3 *u,
                  const float *radius, const bool *inside,
                  const uint *sample_offsets, const uint *sample_ids,
                  SdfDcOptions opt, float tol) {
    uint i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (i >= num_cells) {
        return;
    }
    uint s_begin = sample_offsets[i], s_end = sample_offsets[i + 1];
    if (s_begin == s_end) {
        return;
    }

    // The cell's Hermite data and local mesh: triangles
    // (vertex, Hermite point k, face point d).
    uint m_begin = its_offsets[i], m_end = its_offsets[i + 1];
    float3 hp[12], hn[12];
    int hedge[12];
    int num_h = 0;
    for (uint m = m_begin; m < m_end && num_h < 12; m++) {
        hp[num_h] = h[edge_of[m]];
        hn[num_h] = n[edge_of[m]];
        hedge[num_h] = local_edge[m];
        num_h++;
    }
    int tri_k[24], tri_d[24];
    int num_tris = 0;
    for (int d = 0; d < 6; d++) {
        if (isnan(p[i * 6 + d].x)) {
            continue;
        }
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < num_h; j++) {
                if (hedge[j] == face_edge(d, k)) {
                    tri_k[num_tris] = j;
                    tri_d[num_tris] = d;
                    num_tris++;
                }
            }
        }
    }
    if (num_tris == 0) {
        return;
    }

    float3 xc = x[i];
    for (int iter = 0; iter < opt.inner_iters; iter++) {
        float3 xp = xc;
        float ATA[3][3] = {};
        float3 ATb = make_float3(0.0f, 0.0f, 0.0f);
        auto add_row = [&](float3 a, float b) {
            ATA[0][0] += a.x * a.x;
            ATA[0][1] += a.x * a.y;
            ATA[0][2] += a.x * a.z;
            ATA[1][1] += a.y * a.y;
            ATA[1][2] += a.y * a.z;
            ATA[2][2] += a.z * a.z;
            ATb = ATb + a * b;
        };

        // Hermite planes (Eq. 5), one per lane.
        if (lane < num_h) {
            float3 a = hn[lane] * opt.hermite_weight;
            add_row(a, dot(a, hp[lane]));
        }

        // Sphere tangency, linearized along the radial direction
        // (Eq. 9, 10). The reference keeps d = q - c unnormalized, so rows
        // are weighted by the sphere radius; the paper's unit d was much
        // less accurate in our runs.
        for (uint s = s_begin + lane; s < s_end; s += 32) {
            uint j = sample_ids[s];
            float3 c = u[j];
            float best = FMAX;
            float3 bary = make_float3(1.0f, 0.0f, 0.0f);
            int best_t = 0;
            for (int t = 0; t < num_tris; t++) {
                float3 a = hp[tri_k[t]], b = p[i * 6 + tri_d[t]];
                float3 q = closest_point_triangle(c, xp, a, b);
                float d2 = dot(q - c, q - c);
                if (d2 < best) {
                    best = d2;
                    best_t = t;
                    bary = barycentric(q, xp, a, b);
                }
            }
            float3 hk = hp[tri_k[best_t]], pd = p[i * 6 + tri_d[best_t]];
            // A closest point off the vertex cannot move it; let the vertex
            // itself approach the sphere instead (Appendix C).
            if (fabsf(bary.x) < 1e-6f) {
                bary = make_float3(1.0f, 0.0f, 0.0f);
            }
            float3 tpt = xp * bary.x + hk * bary.y + pd * bary.z;
            float3 v = tpt - c;
            float rho = norm(v);
            if (rho > radius[j] && inside[j]) {
                continue;
            }
            float3 q = rho < 1e-9f ? c : c + v * (radius[j] / rho);
            float3 d = q - c;
            add_row(d * bary.x,
                    dot(q, d) - bary.y * dot(hk, d) - bary.z * dot(pd, d));
        }

        ATA[0][0] = warp_sum(ATA[0][0]);
        ATA[0][1] = warp_sum(ATA[0][1]);
        ATA[0][2] = warp_sum(ATA[0][2]);
        ATA[1][1] = warp_sum(ATA[1][1]);
        ATA[1][2] = warp_sum(ATA[1][2]);
        ATA[2][2] = warp_sum(ATA[2][2]);
        ATb = warp_sum(ATb);

        // Step regularization toward the previous iterate.
        ATA[0][0] += opt.mu;
        ATA[1][1] += opt.mu;
        ATA[2][2] += opt.mu;
        ATb = ATb + xp * opt.mu;
        ATA[1][0] = ATA[0][1];
        ATA[2][0] = ATA[0][2];
        ATA[2][1] = ATA[1][2];

        // mu > 0 keeps the system positive definite, so this is a plain
        // solve; the reference's pseudo-inverse truncation never engages.
        xc = solve_sym_3x3(ATA, ATb, 1e-6f);
        if (norm(xc - xp) < tol) {
            break;
        }
    }
    if (lane == 0) {
        x[i] = xc;
    }
}

}   // anonymous namespace

std::pair<NDArray<float3>, NDArray<int>>
dual_contouring_sdf(Grid *grid, const Intersection &its, float level,
                    const SdfDcOptions &opt) {
    uint num_cells = its.cell_indices.size();
    if (num_cells == 0) {
        return {NDArray<float3>({0}), NDArray<int>({0, 3})};
    }
    GridView view = grid->get_view();
    CellLookup lookup{view, grid->get_num_cells(), its.cell_indices.data(),
                      num_cells};
    float3 cell_size =
        (view.aabb_max - view.aabb_min) / to_float3(view.shape - 1);
    float diag = norm(cell_size);
    float h_min = fminf(cell_size.x, fminf(cell_size.y, cell_size.z));

    static const thrust::device_vector<int> &edges_tab =
        *new thrust::device_vector<int>(edges_table, edges_table + edges_size);

    // Unique crossed edges, their four cells as dual indices, and the
    // Hermite data seeded from the crossings.
    auto [quads_dv, is_out_dv, unique_edges] =
        grid->get_dual_quads(its.edges, its.is_out);
    uint num_edges = unique_edges.size();
    uint num_crossings = its.points.size();
    thrust::device_vector<int4> quads(num_edges);
    thrust::transform(quads_dv.begin(), quads_dv.end(), quads.begin(),
                      [lookup] __device__(int4 q) {
                          auto to_dual = [&](int cell) {
                              return cell < 0
                                         ? -1
                                         : lookup.find(lookup.its_cells,
                                                       lookup.num_its_cells,
                                                       uint(cell));
                          };
                          return make_int4(to_dual(q.x), to_dual(q.y),
                                           to_dual(q.z), to_dual(q.w));
                      });
    thrust::device_vector<float3> h(num_edges), n(num_edges);
    thrust::device_vector<float3> g(num_crossings);
    thrust::device_vector<float3> edge_a(num_edges), edge_b(num_edges);
    thrust::device_vector<uint8_t> local_edge(num_crossings);
    thrust::device_vector<uint> edge_of(num_crossings);
    thrust::device_vector<float3> x(num_cells);
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_cells),
        prep_cells_op{local_edge.data().get(), edge_of.data().get(),
                      g.data().get(), h.data().get(), edge_a.data().get(),
                      edge_b.data().get(), x.data().get(), its.points.data(),
                      its.has_normals() ? its.normals.data() : nullptr,
                      its.edges.data(), its.cell_offsets.data(),
                      its.cell_indices.data(), unique_edges.data().get(),
                      num_edges, view, edges_tab.data().get()});
    {
        // Every unique edge has at least one crossing, so the segmented
        // sum lands one entry per edge, in edge order.
        thrust::device_vector<uint> keys = edge_of;
        thrust::sort_by_key(keys.begin(), keys.end(), g.begin());
        thrust::device_vector<uint> out_keys(num_edges);
        thrust::reduce_by_key(
            keys.begin(), keys.end(), g.begin(), out_keys.begin(), n.begin(),
            thrust::equal_to<uint>(),
            [] __device__(float3 a, float3 b) { return a + b; });
        thrust::transform(
            n.begin(), n.end(), n.begin(), [] __device__(float3 v) {
                return normalize_or(v, make_float3(0.0f, 0.0f, 1.0f));
            });
    }

    // Global mesh triangles: the edges with all four cells present.
    TriangleSoup soup;
    {
        thrust::device_vector<uint> full(num_edges);
        auto end = thrust::copy_if(
            thrust::counting_iterator<uint>(0),
            thrust::counting_iterator<uint>(num_edges), quads.begin(),
            full.begin(), [] __device__(int4 q) {
                return q.x >= 0 && q.y >= 0 && q.z >= 0 && q.w >= 0;
            });
        full.erase(end, full.end());
        soup.tris.resize(2 * full.size());
        thrust::for_each(thrust::counting_iterator<uint>(0),
                         thrust::counting_iterator<uint>(full.size()),
                         [full = full.data().get(), quads = quads.data().get(),
                          tris = soup.tris.data().get()] __device__(uint k) {
                             int4 q = quads[full[k]];
                             tris[2 * k] = make_uint3(q.x, q.y, q.z);
                             tris[2 * k + 1] = make_uint3(q.x, q.z, q.w);
                         });
    }

    Samples samples =
        gather_samples(view, grid->get_num_cells(), level, opt.band * diag);
    uint num_samples = samples.u.size();
    thrust::device_vector<int> cell_of(num_samples);
    thrust::device_vector<uint> sample_ids(num_samples);
    thrust::device_vector<uint> sample_offsets(num_cells + 1);
    thrust::device_vector<float3> p(num_cells * 6, make_float3(NAN, NAN, NAN));
    float tol = opt.tol * diag;

    thrust::device_vector<float3> x_qef;
    if (opt.qef_assignment) {
        x_qef.resize(num_cells);
        thrust::for_each(thrust::counting_iterator<uint>(0),
                         thrust::counting_iterator<uint>(num_cells),
                         qef_vertices_op{x_qef.data().get(), x.data().get(),
                                         h.data().get(), n.data().get(),
                                         edge_of.data().get(),
                                         its.cell_offsets.data()});
    }

    for (int outer = 0; outer < opt.outer_iters; outer++) {
        // Assign every sample to the cell holding its closest mesh point.
        const thrust::device_vector<float3> &mesh_x =
            (opt.qef_assignment && outer == 0) ? x_qef : x;
        bin_triangles(soup, mesh_x, lookup);
        if (num_samples > 0) {
            assign_samples_kernel<<<(num_samples * 32 + 255) / 256, 256>>>(
                cell_of.data().get(), samples.u.data().get(),
                samples.radius.data().get(), num_samples,
                soup.tris.data().get(), mesh_x.data().get(),
                soup.bin_keys.data().get(), soup.bin_tris.data().get(),
                uint(soup.bin_keys.size()), soup.occupied.data().get(), lookup,
                diag, h_min);
        }
        thrust::device_vector<int> keys = cell_of;
        thrust::sequence(sample_ids.begin(), sample_ids.end());
        thrust::sort_by_key(keys.begin(), keys.end(), sample_ids.begin());
        thrust::lower_bound(keys.begin(), keys.end(),
                            thrust::counting_iterator<int>(0),
                            thrust::counting_iterator<int>(num_cells + 1),
                            sample_offsets.begin());

        thrust::for_each(thrust::counting_iterator<uint>(0),
                         thrust::counting_iterator<uint>(num_cells),
                         face_points_op{p.data().get(), x.data().get(), lookup,
                                        opt.update_weight});
        if (opt.hermite_update && outer > 0) {
            thrust::for_each(
                thrust::counting_iterator<uint>(0),
                thrust::counting_iterator<uint>(num_edges),
                hermite_update_op{h.data().get(), n.data().get(),
                                  x.data().get(), quads.data().get(),
                                  edge_a.data().get(), edge_b.data().get(),
                                  is_out_dv.data().get(), opt.update_weight});
        }
        inner_loop_kernel<<<(num_cells * 32 + 255) / 256, 256>>>(
            x.data().get(), p.data().get(), h.data().get(), n.data().get(),
            edge_of.data().get(), local_edge.data().get(),
            its.cell_offsets.data(), num_cells, samples.u.data().get(),
            samples.radius.data().get(), samples.inside.data().get(),
            sample_offsets.data().get(), sample_ids.data().get(), opt, tol);
    }

    return build_dual_mesh(grid, its, x);
}
