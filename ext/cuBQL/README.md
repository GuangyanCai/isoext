# cuBQL (vendored subset)

The headers under `cuBQL/` are the subset of NVIDIA's cuBQL
(https://github.com/NVIDIA/cuBQL, version 1.3.1, commit 20f9db19cdb160e2b83d8f05dfdb0437fe11fe24) that
isoext needs for the mesh SDF: the BVH types and GPU builder, the closest
point on a triangle mesh query, and the inside/outside test. They are
unmodified. cuBQL is licensed under the Apache License 2.0; see LICENSE.
