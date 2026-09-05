#include "dc.cuh"
#include "dmc.cuh"
#include "extraction.cuh"
#include "mc/mc.cuh"
#include "shared_luts.cuh"
#include "sym3x3.cuh"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

// Dual marching cubes, after Nielson, "Dual Marching Cubes" (2004): the
// dual of the marching cubes mesh.
//
// The selected marching cubes variant triangulates each crossed cell, and
// the connected components of that triangulation are the cell's patches:
// most cells have one, cells crossed by several surface sheets have more.
// Each patch becomes one vertex, placed from the patch's own edge
// crossings: the QEF minimizer of their tangent planes when the
// intersection carries normals (sharp features, like dual contouring),
// their centroid otherwise. Every unique crossed grid edge then yields a
// quad connecting the patch vertices of the four cells around it -- in
// each of those cells, the patch that contains that edge's crossing.
//
// Because a cell can host one vertex per patch, configurations that break
// dual contouring's one-vertex-per-cell connectivity (two sheets through
// one cell) come out clean, and the topology follows the chosen variant's
// tables. Edges on the grid boundary have fewer than four adjacent cells
// and are skipped, so the mesh ends half a cell short of the boundary.

namespace {

// Twelve edges and at least three patch corners each bound the number of
// connected components of a cell's triangulation.
constexpr uint max_patches = 4;
constexpr uint max_tri = 12;

constexpr float qef_reg = 1e-2f;
constexpr float qef_tol = 1e-6f;

// One thread per active cell: group the variant's triangles into patches,
// map each crossed local edge to its patch, and place one vertex per patch.
struct analyze_cells_op {
    float3 *patch_verts;      // num_active * max_patches
    int8_t *edge_patch;       // num_active * 12; -1 when the edge is unused
    const float3 *soup;       // num_active * soup_stride, NAN-padded
    const uint soup_stride;   // the variant's max_triangles * 3
    const float3 *its_points;
    const float3 *its_normals;   // null without normals
    const uint2 *its_edges;
    const uint *cell_offsets;
    const uint *active_cells;
    const GridView view;
    const int *edges_tab;

    analyze_cells_op(float3 *patch_verts, int8_t *edge_patch,
                     const float3 *soup, uint soup_stride,
                     const float3 *its_points, const float3 *its_normals,
                     const uint2 *its_edges, const uint *cell_offsets,
                     const uint *active_cells, const GridView &view,
                     const int *edges_tab)
        : patch_verts(patch_verts), edge_patch(edge_patch), soup(soup),
          soup_stride(soup_stride), its_points(its_points),
          its_normals(its_normals), its_edges(its_edges),
          cell_offsets(cell_offsets), active_cells(active_cells), view(view),
          edges_tab(edges_tab) {}

    __host__ __device__ void operator()(uint idx) {
        uint cell = active_cells[idx];
        float3 c_p[8];
        for (int i = 0; i < 8; i++) {
            c_p[i] = view.corner_position(cell, i);
        }

        // The variant wrote its triangles contiguously; NAN ends the list.
        const float3 *v = soup + idx * soup_stride;
        int nt = 0;
        while (nt < int(soup_stride / 3) && !isnan(v[nt * 3].x)) {
            nt++;
        }

        // Group triangles into patches: union by shared vertex positions,
        // which are bitwise equal within a cell.
        int parent[max_tri];
        for (int t = 0; t < nt; t++) {
            parent[t] = t;
        }
        auto find = [&](int t) {
            while (parent[t] != t) {
                parent[t] = parent[parent[t]];
                t = parent[t];
            }
            return t;
        };
        for (int t1 = 1; t1 < nt; t1++) {
            for (int t0 = 0; t0 < t1; t0++) {
                bool shared = false;
                for (int i = 0; i < 3 && !shared; i++) {
                    for (int j = 0; j < 3 && !shared; j++) {
                        shared = v[t1 * 3 + i] == v[t0 * 3 + j];
                    }
                }
                if (shared) {
                    parent[find(t1)] = find(t0);
                }
            }
        }
        int patch_of_root[max_tri];
        int num_patches = 0;
        for (int t = 0; t < nt; t++) {
            if (find(t) == t) {
                patch_of_root[t] = num_patches++;
            }
        }

        // Assign each crossed local edge to the patch whose triangle has a
        // vertex on it. Interior vertices (the MC33 center vertex) lie on
        // no edge; they only connect triangles into one patch.
        int8_t *ep = edge_patch + idx * 12;
        for (int e = 0; e < 12; e++) {
            ep[e] = -1;
        }
        for (int k = 0; k < nt * 3; k++) {
            float3 p = v[k];
            // The closest edge wins: a vertex near a corner is within
            // tolerance of all three edges through it, and picking the
            // wrong one would leave the right edge without a patch.
            int best = -1;
            float best_ratio = 1e-6f;
            for (int e = 0; e < 12; e++) {
                float3 a = c_p[edges_tab[2 * e]];
                float3 b = c_p[edges_tab[2 * e + 1]];
                float3 d = b - a;
                float3 u = p - a;
                float dd = dot(d, d);
                float t = dot(u, d) / dd;
                if (t < -1e-4f || t > 1.0001f) {
                    continue;
                }
                float3 perp = u - t * d;
                float ratio = dot(perp, perp) / dd;
                if (ratio < best_ratio) {
                    best_ratio = ratio;
                    best = e;
                }
            }
            if (best >= 0) {
                ep[best] = int8_t(patch_of_root[find(k / 3)]);
            }
        }

        // Place one vertex per patch from the patch's crossings: the QEF
        // of their tangent planes when normals are available (the same
        // math as place_dual_vertex_op in dc.cu), their centroid
        // otherwise.
        float ata[max_patches][6] = {};
        float3 atb[max_patches];
        float3 centroid[max_patches];
        int count[max_patches] = {};
        for (uint p = 0; p < max_patches; p++) {
            atb[p] = make_float3(0.0f, 0.0f, 0.0f);
            centroid[p] = make_float3(0.0f, 0.0f, 0.0f);
        }

        for (uint j = cell_offsets[idx]; j < cell_offsets[idx + 1]; j++) {
            // The crossing's local edge, from its endpoint ids.
            uint2 edge = its_edges[j];
            int ca = -1, cb = -1;
            for (int c = 0; c < 8; c++) {
                uint id = view.corner_point_id(cell, c);
                if (id == edge.x) {
                    ca = c;
                }
                if (id == edge.y) {
                    cb = c;
                }
            }
            int e = -1;
            for (int i = 0; i < 12 && e < 0; i++) {
                int p0 = edges_tab[2 * i], p1 = edges_tab[2 * i + 1];
                if ((p0 == ca && p1 == cb) || (p0 == cb && p1 == ca)) {
                    e = i;
                }
            }
            int p = (e >= 0) ? ep[e] : -1;
            if (p < 0) {
                continue;
            }

            float3 x = its_points[j];
            centroid[p] = centroid[p] + x;
            count[p]++;
            if (its_normals) {
                float3 n = its_normals[j];
                ata[p][0] += n.x * n.x;
                ata[p][1] += n.x * n.y;
                ata[p][2] += n.x * n.z;
                ata[p][3] += n.y * n.y;
                ata[p][4] += n.y * n.z;
                ata[p][5] += n.z * n.z;
                atb[p] = atb[p] + n * dot(n, x);
            }
        }

        for (int p = 0; p < num_patches; p++) {
            float3 x;
            if (count[p] == 0) {
                // Tie conventions can disagree about a crossed edge; fall
                // back to the cell center.
                x = 0.5f * (c_p[0] + c_p[7]);
            } else if (its_normals) {
                float3 mean = centroid[p] / float(count[p]);
                float m[3][3] = {
                    {ata[p][0] + qef_reg, ata[p][1], ata[p][2]},
                    {ata[p][1], ata[p][3] + qef_reg, ata[p][4]},
                    {ata[p][2], ata[p][4], ata[p][5] + qef_reg},
                };
                float3 rhs = atb[p] + qef_reg * mean;
                x = clip(solve_sym_3x3(m, rhs, qef_tol), c_p[0], c_p[7]);
            } else {
                x = centroid[p] / float(count[p]);
            }
            patch_verts[idx * max_patches + p] = x;
        }
    }
};

// One thread per unique crossed edge: connect the patch vertices of the
// four adjacent cells into a quad (two triangles).
struct emit_quads_op {
    float3 *out;   // num_quads * 6
    const int4 *quads;
    const bool *is_out;
    const uint2 *quad_edges;   // dense lattice point ids
    const float3 *patch_verts;
    const int8_t *edge_patch;
    const uint *active_cells;   // sorted
    const uint num_active;
    const GridView view;
    const int *edges_tab;

    emit_quads_op(float3 *out, const int4 *quads, const bool *is_out,
                  const uint2 *quad_edges, const float3 *patch_verts,
                  const int8_t *edge_patch, const uint *active_cells,
                  uint num_active, const GridView &view, const int *edges_tab)
        : out(out), quads(quads), is_out(is_out), quad_edges(quad_edges),
          patch_verts(patch_verts), edge_patch(edge_patch),
          active_cells(active_cells), num_active(num_active), view(view),
          edges_tab(edges_tab) {}

    // The patch vertex of `cell` for the crossing on `edge`.
    __host__ __device__ bool vertex_for(int cell, uint2 edge,
                                        float3 &result) const {
        const uint *end = active_cells + num_active;
        const uint *it =
            thrust::lower_bound(thrust::seq, active_cells, end, uint(cell));
        if (it == end || *it != uint(cell)) {
            return false;
        }
        uint slot = it - active_cells;

        int ca = -1, cb = -1;
        for (int c = 0; c < 8; c++) {
            uint id = view.dense_point_index(uint(cell), c);
            if (id == edge.x) {
                ca = c;
            }
            if (id == edge.y) {
                cb = c;
            }
        }
        int e = -1;
        for (int i = 0; i < 12 && e < 0; i++) {
            int p0 = edges_tab[2 * i], p1 = edges_tab[2 * i + 1];
            if ((p0 == ca && p1 == cb) || (p0 == cb && p1 == ca)) {
                e = i;
            }
        }
        if (e < 0) {
            return false;
        }
        int p = edge_patch[slot * 12 + e];
        if (p < 0) {
            return false;
        }
        result = patch_verts[slot * max_patches + p];
        return true;
    }

    __host__ __device__ void operator()(uint idx) {
        int4 q = quads[idx];
        if (q.x == -1 || q.y == -1 || q.z == -1 || q.w == -1) {
            return;
        }
        uint2 edge = quad_edges[idx];

        float3 v0, v1, v2, v3;
        if (!vertex_for(q.x, edge, v0) || !vertex_for(q.y, edge, v1) ||
            !vertex_for(q.z, edge, v2) || !vertex_for(q.w, edge, v3)) {
            return;
        }

        // An inward edge reverses the quad, same as dual contouring.
        if (!is_out[idx]) {
            float3 tmp = v1;
            v1 = v3;
            v3 = tmp;
        }

        // Split along the shorter diagonal.
        float3 *o = out + idx * 6;
        if (norm(v0 - v2) > norm(v1 - v3)) {
            if (v1 != v3 && v1 != v0 && v3 != v0) {
                o[0] = v1;
                o[1] = v3;
                o[2] = v0;
            }
            if (v3 != v1 && v3 != v2 && v1 != v2) {
                o[3] = v3;
                o[4] = v1;
                o[5] = v2;
            }
        } else {
            if (v2 != v0 && v2 != v1 && v0 != v1) {
                o[0] = v2;
                o[1] = v0;
                o[2] = v1;
            }
            if (v0 != v2 && v0 != v3 && v2 != v3) {
                o[3] = v0;
                o[4] = v2;
                o[5] = v3;
            }
        }
    }
};

}   // anonymous namespace

std::tuple<NDArray<float3>, NDArray<int>>
dual_marching_cubes(Grid *grid, const Intersection &its, float level,
                    std::string method) {
    auto mc_variant = mc::MCBase::create(method);
    GridView view = grid->get_view();

    uint num_active = its.cell_indices.size();
    if (num_active == 0) {
        return {NDArray<float3>({0}), NDArray<int>({0, 3})};
    }
    if (mc_variant->get_max_triangles() > max_tri) {
        throw std::runtime_error(
            "the variant's triangle budget exceeds the dual mesh buffer");
    }

    static const thrust::device_vector<int> &edges_tab =
        *new thrust::device_vector<int>(edges_table, edges_table + edges_size);

    // 1. The variant's triangulation of every active cell. The variants
    // expect the case array compacted to the active cells.
    thrust::device_vector<uint8_t> cases_full =
        compute_cell_cases(view, grid->get_num_cells(), level);
    thrust::device_vector<uint8_t> cases(num_active);
    thrust::gather(thrust::device, its.cell_indices.data(),
                   its.cell_indices.data() + num_active, cases_full.begin(),
                   cases.begin());
    uint soup_stride = mc_variant->get_max_triangles() * 3;
    thrust::device_vector<float3> soup(num_active * soup_stride);
    thrust::fill(soup.begin(), soup.end(), make_float3(NAN, NAN, NAN));
    mc_variant->run(soup.data().get(), num_active, cases.data().get(),
                    its.cell_indices.data(), view, level);

    // 2. Patches and their vertices.
    thrust::device_vector<float3> patch_verts(num_active * max_patches);
    thrust::device_vector<int8_t> edge_patch(num_active * 12);
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_active),
        analyze_cells_op(patch_verts.data().get(), edge_patch.data().get(),
                         soup.data().get(), soup_stride, its.points.data(),
                         its.has_normals() ? its.normals.data() : nullptr,
                         its.edges.data(), its.cell_offsets.data(),
                         its.cell_indices.data(), view,
                         edges_tab.data().get()));

    // 3. One quad per unique crossed edge.
    auto [quads, quad_is_out, quad_edges] =
        grid->get_dual_quads(its.edges, its.is_out);
    uint num_quads = quads.size();
    thrust::device_vector<float3> v(num_quads * 6, make_float3(NAN, NAN, NAN));
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_quads),
        emit_quads_op(v.data().get(), quads.data().get(),
                      quad_is_out.data().get(), quad_edges.data().get(),
                      patch_verts.data().get(), edge_patch.data().get(),
                      its.cell_indices.data(), num_active, view,
                      edges_tab.data().get()));

    auto [v_out, f_out] = soup_to_mesh(v);
    return {v_out, f_out};
}
