#include "batched_la.cuh"
#include "dc.cuh"
#include "math.cuh"
#include "utils.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <tuple>

namespace {

struct get_qef_op {
    float *ATA;
    float *ATb;
    const float3 *its_points;
    const float3 *its_normals;
    const uint *its_cell_offsets;
    const float reg;

    get_qef_op(float *ATA, float *ATb, const float3 *its_points,
               const float3 *its_normals, const uint *cell_offsets, float reg)
        : ATA(ATA), ATb(ATb), its_points(its_points), its_normals(its_normals),
          its_cell_offsets(cell_offsets), reg(reg) {}

    __host__ __device__ void operator()(uint idx) {
        uint ATA_offset = idx * 9;
        uint ATb_offset = idx * 3;
        float3 p_avg = make_float3(0.0f, 0.0f, 0.0f);
        for (uint i = its_cell_offsets[idx]; i < its_cell_offsets[idx + 1];
             i++) {
            float3 n = its_normals[i];
            float3 p = its_points[i];
            float3 nnp = n * dot(n, p);
            p_avg = p_avg + p;

            // ATA
            ATA[ATA_offset + 0] += n.x * n.x;
            ATA[ATA_offset + 1] += n.x * n.y;
            ATA[ATA_offset + 2] += n.x * n.z;
            ATA[ATA_offset + 3] += n.y * n.x;
            ATA[ATA_offset + 4] += n.y * n.y;
            ATA[ATA_offset + 5] += n.y * n.z;
            ATA[ATA_offset + 6] += n.z * n.x;
            ATA[ATA_offset + 7] += n.z * n.y;
            ATA[ATA_offset + 8] += n.z * n.z;

            // ATb
            ATb[ATb_offset + 0] += nnp.x;
            ATb[ATb_offset + 1] += nnp.y;
            ATb[ATb_offset + 2] += nnp.z;
        }
        p_avg = p_avg / (its_cell_offsets[idx + 1] - its_cell_offsets[idx]);

        // Add λI to ATA
        ATA[ATA_offset + 0] += reg;
        ATA[ATA_offset + 4] += reg;
        ATA[ATA_offset + 8] += reg;

        ATb[ATb_offset + 0] += reg * p_avg.x;
        ATb[ATb_offset + 1] += reg * p_avg.y;
        ATb[ATb_offset + 2] += reg * p_avg.z;
    }
};

std::tuple<NDArray<float>, NDArray<float>>
get_qef(const Intersection &its, float reg) {
    uint num_cells = its.cell_indices.size();
    NDArray<float> ATA = NDArray<float>::zeros(num_cells, 3, 3);
    NDArray<float> ATb = NDArray<float>::zeros(num_cells, 3);

    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_cells),
                     get_qef_op(ATA.data(), ATb.data(), its.points.data(),
                                its.normals.data(), its.cell_offsets.data(),
                                reg));

    return {ATA, ATb};
}

struct fix_dual_v_op {
    float3 *dual_v;
    const uint *its_cell_indices;
    const float3 *points;
    const uint *cells;

    fix_dual_v_op(float3 *dual_v, const uint *its_cell_indices,
                  const float3 *points, const uint *cells)
        : dual_v(dual_v), its_cell_indices(its_cell_indices), points(points),
          cells(cells) {}

    __host__ __device__ void operator()(uint idx) {
        uint offset = its_cell_indices[idx] * 8;
        float3 aabb_min = points[cells[offset]];
        float3 aabb_max = points[cells[offset + 7]];
        dual_v[idx] = clip(dual_v[idx], aabb_min, aabb_max);
    }
};

struct get_triangles_op {
    float3 *v;
    const float3 *dual_v;
    const int4 *quad_indices;
    const uint2 *its_edges;
    const float *values;
    const int *idx_map;

    get_triangles_op(float3 *v, const float3 *dual_v, const int4 *quad_indices,
                     const uint2 *its_edges, const float *values,
                     const int *idx_map)
        : v(v), dual_v(dual_v), quad_indices(quad_indices),
          its_edges(its_edges), values(values), idx_map(idx_map) {}

    __host__ __device__ void operator()(uint idx) {
        int4 quad_idx = quad_indices[idx];
        if (quad_idx.x == -1 || quad_idx.y == -1 || quad_idx.z == -1 ||
            quad_idx.w == -1) {
            return;
        }

        quad_idx.x = idx_map[quad_idx.x];
        quad_idx.y = idx_map[quad_idx.y];
        quad_idx.z = idx_map[quad_idx.z];
        quad_idx.w = idx_map[quad_idx.w];

        // If the edge is pointing inward, swap the quad indices.
        uint2 edge = its_edges[idx];
        if (values[edge.x] > values[edge.y]) {
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

}   // anonymous namespace

std::pair<NDArray<float3>, NDArray<int>>
dual_contouring(Grid *grid, const Intersection &its, float level, float reg,
                float svd_tol) {
    thrust::device_vector<uint2> edges_dv(its.edges.data(),
                                          its.edges.data() + its.edges.size());

    // Sort edges to bring duplicates together
    thrust::sort(edges_dv.begin(), edges_dv.end(), uint2_less_pred());

    // Remove duplicates
    edges_dv.erase(
        thrust::unique(edges_dv.begin(), edges_dv.end(), uint2_equal_pred()),
        edges_dv.end());

    uint num_edges = edges_dv.size();

    // Get edge neighbors
    uint3 shape = grid->get_shape();
    thrust::device_vector<int4> edge_neighbors =
        get_edge_neighbors(edges_dv, shape);

    auto [ATA, ATb] = get_qef(its, reg);
    BatchedLASolver solver;
    auto [dual_v, info] = solver.lsq_svd(ATA, ATb, svd_tol);

    // Clip dual vertices to the cell AABB
    NDArray<uint> cells = grid->get_cells();
    NDArray<float3> points = grid->get_points();
    uint num_active_cells = its.cell_indices.size();
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_active_cells),
                     fix_dual_v_op(reinterpret_cast<float3 *>(dual_v.data()),
                                   its.cell_indices.data(), points.data(),
                                   cells.data()));
    cells.free();
    points.free();

    // Create index map that maps cell indices to dual_v indices
    thrust::device_vector<int> idx_map(grid->get_num_cells(), -1);
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(its.cell_indices.size()),
        [idx_map = idx_map.data(),
         cell_indices = its.cell_indices.data()] __device__(uint i) {
            idx_map[cell_indices[i]] = i;
        });

    const NDArray<float> &values = grid->get_values();
    thrust::device_vector<float3> v_dv(num_edges * 6,
                                       make_float3(NAN, NAN, NAN));
    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(num_edges),
                     get_triangles_op(v_dv.data().get(),
                                      reinterpret_cast<float3 *>(dual_v.data()),
                                      edge_neighbors.data().get(),
                                      edges_dv.data().get(), values.data(),
                                      idx_map.data().get()));

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
