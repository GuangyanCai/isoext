#include "mesh_sdf.cuh"

// Pull the cuBQL GPU builder into this translation unit only. bvh.h must
// come first, so the include order is kept out of clang-format's hands.
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
// clang-format off
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda/refit.h"
#include "cuBQL/queries/triangleData/closestPointOnAnyTriangle.h"
#include "cuBQL/queries/triangleData/math/rayTriangleIntersections.h"
#include "cuBQL/traversal/rayQueries.h"
// clang-format on

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace {

__host__ __device__ inline cuBQL::vec3f
to_vec(float3 p) {
    return cuBQL::vec3f(p.x, p.y, p.z);
}

__host__ __device__ inline float3
to_float3(cuBQL::vec3f p) {
    return make_float3(p.x, p.y, p.z);
}

// Whether a point lies inside a closed mesh, by a majority vote over the
// crossing parities of six rays. The directions are fixed and skewed
// against the axes: rays along grid lines would run exactly through the
// vertices of a marching cubes mesh, and a ray through a vertex or an
// edge counts its triangles twice or not at all. Parity does not depend
// on the mesh orientation.
__device__ inline bool
inside_mesh(const cuBQL::bvh3f &bvh, const cuBQL::Triangle *tris,
            cuBQL::vec3f p) {
    const cuBQL::vec3f dirs[3] = {cuBQL::vec3f(0.8203f, 0.5070f, 0.2646f),
                                  cuBQL::vec3f(0.2646f, 0.8203f, 0.5070f),
                                  cuBQL::vec3f(0.5070f, 0.2646f, 0.8203f)};
    int votes = 0;
    for (int k = 0; k < 6; k++) {
        cuBQL::vec3f dir = (k < 3) ? dirs[k] : dirs[k - 3] * -1.0f;
        cuBQL::Ray ray(p, dir);
        int crossings = 0;
        auto count = [&](uint32_t id) {
            if (cuBQL::rayIntersectsTriangle(ray, tris[id])) {
                crossings++;
            }
            return CUBQL_CONTINUE_TRAVERSAL;
        };
        cuBQL::fixedRayQuery::forEachPrim(count, bvh, ray);
        votes += crossings & 1;
    }
    return votes > 3;
}

// Triangles and their bounding boxes from indexed vertices. A free
// function because extended device lambdas cannot live in constructors.
void
build_triangles(const NDArray<float3> &vertices, const NDArray<int> &faces,
                thrust::device_vector<cuBQL::Triangle> &triangles,
                thrust::device_vector<cuBQL::box3f> &boxes) {
    uint num_faces = faces.shape[0];
    triangles.resize(num_faces);
    boxes.resize(num_faces);
    thrust::for_each(
        thrust::counting_iterator<uint>(0),
        thrust::counting_iterator<uint>(num_faces),
        [tris = triangles.data().get(), boxes = boxes.data().get(),
         v = vertices.data(), f = faces.data()] __device__(uint i) {
            cuBQL::Triangle t(to_vec(v[f[3 * i]]), to_vec(v[f[3 * i + 1]]),
                              to_vec(v[f[3 * i + 2]]));
            tris[i] = t;
            boxes[i] = t.bounds();
        });
}

}   // anonymous namespace

// Per-node data of the fast winding number: the area-weighted normal and
// centroid of the node's triangles, the second moments about that
// centroid, and a radius enclosing the node's vertices around it.
struct WindingNode {
    cuBQL::vec3f normal;   // sum of area * normal
    cuBQL::vec3f center;   // area-weighted centroid
    float area;
    float radius;
    float moments[3][3];   // sum of area * normal_i * (centroid - center)_j
};

__device__ inline WindingNode
winding_leaf(const cuBQL::Triangle *tris, const uint32_t *prim_ids,
             uint32_t count) {
    WindingNode n = {};
    for (uint32_t k = 0; k < count; k++) {
        const cuBQL::Triangle &t = tris[prim_ids[k]];
        cuBQL::vec3f an = 0.5f * cross(t.b - t.a, t.c - t.a);
        float a = length(an);
        n.normal = n.normal + an;
        n.center = n.center + a * (t.a + t.b + t.c) * (1.0f / 3.0f);
        n.area += a;
    }
    if (n.area > 0.0f) {
        n.center = n.center * (1.0f / n.area);
    } else {
        n.center = tris[prim_ids[0]].a;
    }
    for (uint32_t k = 0; k < count; k++) {
        const cuBQL::Triangle &t = tris[prim_ids[k]];
        cuBQL::vec3f an = 0.5f * cross(t.b - t.a, t.c - t.a);
        cuBQL::vec3f d = (t.a + t.b + t.c) * (1.0f / 3.0f) - n.center;
        float ni[3] = {an.x, an.y, an.z}, dj[3] = {d.x, d.y, d.z};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                n.moments[i][j] += ni[i] * dj[j];
            }
        }
        n.radius =
            fmaxf(n.radius,
                  fmaxf(length(t.a - n.center),
                        fmaxf(length(t.b - n.center), length(t.c - n.center))));
    }
    return n;
}

__device__ inline WindingNode
winding_merge(const WindingNode &l, const WindingNode &r) {
    WindingNode n = {};
    n.normal = l.normal + r.normal;
    n.area = l.area + r.area;
    n.center = n.area > 0.0f
                   ? (l.area * l.center + r.area * r.center) * (1.0f / n.area)
                   : 0.5f * (l.center + r.center);
    const WindingNode *kids[2] = {&l, &r};
    for (const WindingNode *k : kids) {
        // Moments shift with the centroid: sum a n (c - p) =
        // sum a n (c - p_k) + (sum a n)(p_k - p).
        cuBQL::vec3f d = k->center - n.center;
        float ni[3] = {k->normal.x, k->normal.y, k->normal.z};
        float dj[3] = {d.x, d.y, d.z};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                n.moments[i][j] += k->moments[i][j] + ni[i] * dj[j];
            }
        }
        n.radius = fmaxf(n.radius, length(d) + k->radius);
    }
    return n;
}

// Bottom-up pass over the BVH, one thread per leaf: the last of a node's
// two children to finish computes the parent (cuBQL's refit scheme;
// refit_data holds parent << 1 with the arrival count in the low bit, so
// the grandparent is read from the value the atomic returns, before the
// second arrival disturbs it).
__global__ void
winding_aggregate_kernel(cuBQL::bvh3f bvh, const cuBQL::Triangle *tris,
                         WindingNode *nodes, uint32_t *refit_data) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id == 1 || id >= int(bvh.numNodes) || bvh.nodes[id].admin.count == 0) {
        return;
    }
    nodes[id] = winding_leaf(tris, bvh.primIDs + bvh.nodes[id].admin.offset,
                             bvh.nodes[id].admin.count);
    __threadfence();
    int parent = refit_data[id] >> 1;
    while (id != 0) {
        uint32_t old = atomicAdd(&refit_data[parent], 1u);
        if ((old & 1) == 0) {
            return;   // the sibling arrives later and continues upward
        }
        int child = bvh.nodes[parent].admin.offset;
        nodes[parent] = winding_merge(nodes[child], nodes[child + 1]);
        __threadfence();
        id = parent;
        parent = old >> 1;
    }
}

// Solid angle of a triangle seen from the origin (Van Oosterom and
// Strackee 1983), signed by the triangle's orientation.
__device__ inline float
solid_angle(cuBQL::vec3f a, cuBQL::vec3f b, cuBQL::vec3f c) {
    float la = length(a), lb = length(b), lc = length(c);
    float num = dot(a, cross(b, c));
    float den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    return 2.0f * atan2f(num, den);
}

// Generalized winding number at p: exact solid angles for the triangles
// of nearby leaves, the second-order expansion for nodes farther than
// beta times their radius (Barill et al. 2018).
__device__ inline float
winding_number_at(const cuBQL::bvh3f &bvh, const cuBQL::Triangle *tris,
                  const WindingNode *nodes, cuBQL::vec3f p, float beta) {
    const float inv4pi = 0.25f / 3.14159265358979f;
    float w = 0.0f;
    int stack[64];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        int id = stack[--top];
        const WindingNode &n = nodes[id];
        cuBQL::vec3f r = n.center - p;
        float dist = length(r);
        if (dist > beta * n.radius && dist > 0.0f) {
            float d3 = dist * dist * dist;
            float rv[3] = {r.x, r.y, r.z};
            float trace = n.moments[0][0] + n.moments[1][1] + n.moments[2][2];
            float rMr = 0.0f;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    rMr += rv[i] * n.moments[i][j] * rv[j];
                }
            }
            w += inv4pi * (dot(n.normal, r) / d3 + trace / d3 -
                           3.0f * rMr / (d3 * dist * dist));
            continue;
        }
        const auto &admin = bvh.nodes[id].admin;
        if (admin.count > 0) {
            for (uint32_t k = 0; k < admin.count; k++) {
                const cuBQL::Triangle &t = tris[bvh.primIDs[admin.offset + k]];
                w += inv4pi * solid_angle(t.a - p, t.b - p, t.c - p);
            }
        } else if (top + 2 <= 64) {
            stack[top++] = admin.offset;
            stack[top++] = admin.offset + 1;
        }
    }
    return w;
}

struct MeshBVH::Impl {
    thrust::device_vector<cuBQL::Triangle> triangles;
    cuBQL::bvh3f bvh;
    thrust::device_vector<WindingNode> winding;   // per BVH node
};

MeshBVH::MeshBVH(const NDArray<float3> &vertices, const NDArray<int> &faces)
    : impl(std::make_unique<Impl>()) {
    thrust::device_vector<cuBQL::box3f> boxes;
    build_triangles(vertices, faces, impl->triangles, boxes);
    cuBQL::gpuBuilder(impl->bvh, boxes.data().get(), boxes.size(),
                      cuBQL::BuildConfig());

    // Aggregate the winding number data bottom-up over the tree.
    uint num_nodes = impl->bvh.numNodes;
    impl->winding.resize(num_nodes);
    thrust::device_vector<uint32_t> refit_data(num_nodes, 0);
    cuBQL::cuda::refit_init<float, 3><<<(num_nodes + 1023) / 1024, 1024>>>(
        impl->bvh.nodes, refit_data.data().get(), num_nodes);
    winding_aggregate_kernel<<<(num_nodes + 127) / 128, 128>>>(
        impl->bvh, impl->triangles.data().get(), impl->winding.data().get(),
        refit_data.data().get());
}

MeshBVH::~MeshBVH() {
    // Plain cudaFree, ignoring errors: at interpreter shutdown the CUDA
    // context may already be gone.
    cudaFree(impl->bvh.nodes);
    cudaFree(impl->bvh.primIDs);
}

uint
MeshBVH::num_faces() const {
    return impl->triangles.size();
}

std::tuple<NDArray<float>, NDArray<float3>, NDArray<int>>
MeshBVH::closest(const NDArray<float3> &points) const {
    uint n = points.size();
    NDArray<float> dist({n});
    NDArray<float3> closest({n});
    NDArray<int> tri({n});
    thrust::for_each(
        thrust::counting_iterator<uint>(0), thrust::counting_iterator<uint>(n),
        [p = points.data(), dist = dist.data(), closest = closest.data(),
         tri = tri.data(), tris = impl->triangles.data().get(),
         bvh = impl->bvh] __device__(uint i) {
            cuBQL::triangles::CPAT cpat;
            cpat.runQuery(tris, bvh, to_vec(p[i]));
            dist[i] = sqrtf(cpat.sqrDist);
            closest[i] = to_float3(cpat.P);
            tri[i] = cpat.triangleIdx;
        });
    return {std::move(dist), std::move(closest), std::move(tri)};
}

NDArray<float>
MeshBVH::sign(const NDArray<float3> &points) const {
    uint n = points.size();
    NDArray<float> sign({n});
    thrust::for_each(
        thrust::counting_iterator<uint>(0), thrust::counting_iterator<uint>(n),
        [p = points.data(), sign = sign.data(),
         tris = impl->triangles.data().get(),
         bvh = impl->bvh] __device__(uint i) {
            sign[i] = inside_mesh(bvh, tris, to_vec(p[i])) ? -1.0f : 1.0f;
        });
    return sign;
}

NDArray<float>
MeshBVH::winding_number(const NDArray<float3> &points) const {
    uint n = points.size();
    NDArray<float> w({n});
    thrust::for_each(
        thrust::counting_iterator<uint>(0), thrust::counting_iterator<uint>(n),
        [p = points.data(), w = w.data(), tris = impl->triangles.data().get(),
         nodes = impl->winding.data().get(),
         bvh = impl->bvh] __device__(uint i) {
            w[i] = winding_number_at(bvh, tris, nodes, to_vec(p[i]), 2.0f);
        });
    return w;
}
