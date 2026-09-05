#include "dc.cuh"
#include "math.cuh"
#include "sym3x3.cuh"
#include "utils.cuh"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

#include <tuple>

namespace {

// Accumulate the QEF of one cell from its intersection points and normals,
// solve it, and keep the resulting dual vertex inside the cell.
struct place_dual_vertex_op {
    float3 *dual_v;
    const float3 *its_points;
    const float3 *its_normals;
    const uint *its_cell_offsets;
    const uint *its_cell_indices;
    const GridView view;
    const float reg;
    const float tol;
    const bool clamp;

    place_dual_vertex_op(float3 *dual_v, const float3 *its_points,
                         const float3 *its_normals,
                         const uint *its_cell_offsets,
                         const uint *its_cell_indices, const GridView &view,
                         float reg, float tol, bool clamp)
        : dual_v(dual_v), its_points(its_points), its_normals(its_normals),
          its_cell_offsets(its_cell_offsets),
          its_cell_indices(its_cell_indices), view(view), reg(reg), tol(tol),
          clamp(clamp) {}

    __host__ __device__ void operator()(uint idx) {
        // The QEF minimizes sum_i (n_i . (x - p_i))^2, accumulated as the
        // normal equations A^T A x = A^T b with rows n_i and b_i = n_i . p_i.
        float ATA[3][3] = {};
        float3 ATb = make_float3(0.0f, 0.0f, 0.0f);
        float3 p_avg = make_float3(0.0f, 0.0f, 0.0f);

        uint begin = its_cell_offsets[idx];
        uint end = its_cell_offsets[idx + 1];
        for (uint i = begin; i < end; i++) {
            float3 n = its_normals[i];
            float3 p = its_points[i];
            ATA[0][0] += n.x * n.x;
            ATA[0][1] += n.x * n.y;
            ATA[0][2] += n.x * n.z;
            ATA[1][1] += n.y * n.y;
            ATA[1][2] += n.y * n.z;
            ATA[2][2] += n.z * n.z;
            ATb = ATb + n * dot(n, p);
            p_avg = p_avg + p;
        }
        ATA[1][0] = ATA[0][1];
        ATA[2][0] = ATA[0][2];
        ATA[2][1] = ATA[1][2];
        p_avg = p_avg / float(end - begin);

        // Tikhonov regularization (λI, λ * centroid) pulls the solution
        // toward the centroid of the intersection points.
        ATA[0][0] += reg;
        ATA[1][1] += reg;
        ATA[2][2] += reg;
        ATb = ATb + reg * p_avg;

        float3 x = solve_sym_3x3(ATA, ATb, tol);

        // Clamping keeps the vertex inside its cell, which is safe but
        // rounds features whose feature line runs through a neighboring
        // cell. Without it the vertex stays wherever the QEF puts it, which
        // follows sharp features better but can self-intersect.
        if (clamp) {
            uint cell = its_cell_indices[idx];
            x = clip(x, view.corner_position(cell, 0),
                     view.corner_position(cell, 7));
        }
        dual_v[idx] = x;
    }
};

struct get_triangles_op {
    float3 *v;
    const float3 *dual_v;
    const int4 *quad_indices;
    const bool *its_is_out;
    const uint *its_cell_indices;   // sorted cell indices with intersections
    const uint num_cells;

    get_triangles_op(float3 *v, const float3 *dual_v, const int4 *quad_indices,
                     const bool *its_is_out, const uint *its_cell_indices,
                     const uint num_cells)
        : v(v), dual_v(dual_v), quad_indices(quad_indices),
          its_is_out(its_is_out), its_cell_indices(its_cell_indices),
          num_cells(num_cells) {}

    // Position of a cell in its_cell_indices (which is the index of its dual
    // vertex), or -1 if the cell has no intersections.
    __host__ __device__ int cell_to_dual_idx(int cell) const {
        const uint *end = its_cell_indices + num_cells;
        const uint *it =
            thrust::lower_bound(thrust::seq, its_cell_indices, end, uint(cell));
        return (it != end && *it == uint(cell)) ? int(it - its_cell_indices)
                                                : -1;
    }

    __host__ __device__ void operator()(uint idx) {
        int4 quad_idx = quad_indices[idx];
        if (quad_idx.x == -1 || quad_idx.y == -1 || quad_idx.z == -1 ||
            quad_idx.w == -1) {
            return;
        }

        quad_idx.x = cell_to_dual_idx(quad_idx.x);
        quad_idx.y = cell_to_dual_idx(quad_idx.y);
        quad_idx.z = cell_to_dual_idx(quad_idx.z);
        quad_idx.w = cell_to_dual_idx(quad_idx.w);

        // Every cell around an intersected edge is normally intersected
        // itself; a miss can only come from inconsistent sparse grid values,
        // so skip the quad instead of indexing out of bounds.
        if (quad_idx.x < 0 || quad_idx.y < 0 || quad_idx.z < 0 ||
            quad_idx.w < 0) {
            return;
        }

        // If the edge is pointing inward, swap the quad indices.
        if (!its_is_out[idx]) {
            quad_idx =
                make_int4(quad_idx.w, quad_idx.z, quad_idx.y, quad_idx.x);
        }

        float3 v0 = dual_v[quad_idx.x];
        float3 v1 = dual_v[quad_idx.y];
        float3 v2 = dual_v[quad_idx.z];
        float3 v3 = dual_v[quad_idx.w];

        // 0 3
        // 1 2

        if (norm(v0 - v2) > norm(v1 - v3)) {
            // Split along the edge v1-v3
            v[idx * 6 + 0] = v1;
            v[idx * 6 + 1] = v3;
            v[idx * 6 + 2] = v0;
            v[idx * 6 + 3] = v3;
            v[idx * 6 + 4] = v1;
            v[idx * 6 + 5] = v2;
        } else {
            // Split along the edge v0-v2
            v[idx * 6 + 0] = v2;
            v[idx * 6 + 1] = v0;
            v[idx * 6 + 2] = v1;
            v[idx * 6 + 3] = v0;
            v[idx * 6 + 4] = v2;
            v[idx * 6 + 5] = v3;
        }
    }
};

// Place each cell's vertex at the centroid of its edge intersection
// points (the surface nets rule). The centroid of points on the cell
// boundary always lies inside the cell, so no clipping is needed.
struct place_centroid_vertex_op {
    float3 *dual_v;
    const float3 *its_points;
    const uint *its_cell_offsets;

    place_centroid_vertex_op(float3 *dual_v, const float3 *its_points,
                             const uint *its_cell_offsets)
        : dual_v(dual_v), its_points(its_points),
          its_cell_offsets(its_cell_offsets) {}

    __host__ __device__ void operator()(uint idx) {
        uint begin = its_cell_offsets[idx];
        uint end = its_cell_offsets[idx + 1];
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        for (uint i = begin; i < end; i++) {
            sum = sum + its_points[i];
        }
        dual_v[idx] = sum / float(end - begin);
    }
};

}   // anonymous namespace

// Shared tail of the dual methods: build one quad around every intersected
// edge from the per-cell vertices, split the quads into triangles, and weld
// the result into an indexed mesh.
std::pair<NDArray<float3>, NDArray<int>>
build_dual_mesh(Grid *grid, const Intersection &its,
                const thrust::device_vector<float3> &dual_v) {
    auto [dual_quads_dv, is_out_dv, dedup_edges_dv] =
        grid->get_dual_quads(its.edges, its.is_out);

    uint num_quads = dual_quads_dv.size();
    thrust::device_vector<float3> v_dv(num_quads * 6,
                                       make_float3(NAN, NAN, NAN));
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_quads),
        get_triangles_op(v_dv.data().get(), dual_v.data().get(),
                         dual_quads_dv.data().get(), is_out_dv.data().get(),
                         its.cell_indices.data(), its.cell_indices.size()));

    // Remove unused entries, which are marked as NAN.
    v_dv.erase(thrust::remove_if(v_dv.begin(), v_dv.end(), is_nan_pred()),
               v_dv.end());

    // Weld/merge vertices.
    thrust::device_vector<int> f_dv(v_dv.size());
    thrust::sequence(f_dv.begin(), f_dv.end());
    vertex_welding(v_dv, f_dv);

    NDArray<float3> v = NDArray<float3>::copy(v_dv.data().get(), {v_dv.size()});
    NDArray<int> f =
        NDArray<int>::copy(f_dv.data().get(), {f_dv.size() / 3, 3});

    return {v, f};
}

thrust::device_vector<float3>
place_dual_vertices(Grid *grid, const Intersection &its, float reg,
                    float svd_tol, bool clamp) {
    uint num_active_cells = its.cell_indices.size();
    thrust::device_vector<float3> dual_v(num_active_cells);
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_active_cells),
        place_dual_vertex_op(dual_v.data().get(), its.points.data(),
                             its.normals.data(), its.cell_offsets.data(),
                             its.cell_indices.data(), grid->get_view(), reg,
                             svd_tol, clamp));
    return dual_v;
}

thrust::device_vector<float3>
place_centroid_vertices(const Intersection &its) {
    uint num_active_cells = its.cell_indices.size();
    thrust::device_vector<float3> dual_v(num_active_cells);
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_active_cells),
                     place_centroid_vertex_op(dual_v.data().get(),
                                              its.points.data(),
                                              its.cell_offsets.data()));
    return dual_v;
}

std::pair<NDArray<float3>, NDArray<int>>
dual_contouring(Grid *grid, const Intersection &its, float level, float reg,
                float svd_tol, bool clamp) {
    // No cell intersects the surface: return an empty mesh instead of
    // running the QEF solver on an empty batch.
    if (its.cell_indices.size() == 0) {
        return {NDArray<float3>({0}), NDArray<int>({0, 3})};
    }

    thrust::device_vector<float3> dual_v =
        place_dual_vertices(grid, its, reg, svd_tol, clamp);
    return build_dual_mesh(grid, its, dual_v);
}

std::pair<NDArray<float3>, NDArray<int>>
surface_nets(Grid *grid, const Intersection &its, float level) {
    if (its.cell_indices.size() == 0) {
        return {NDArray<float3>({0}), NDArray<int>({0, 3})};
    }

    thrust::device_vector<float3> dual_v = place_centroid_vertices(its);
    return build_dual_mesh(grid, its, dual_v);
}
