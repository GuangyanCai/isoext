"""Dual contouring front end: dispatches to the two vertex placement variants."""

from .isoext_ext import _dual_contouring_ju, _dual_contouring_sdf

_JU_OPTIONS = {"reg", "svd_tol", "clamp"}
_CARRERA_OPTIONS = {
    "outer_iters",
    "inner_iters",
    "mu",
    "hermite_weight",
    "update_weight",
    "hermite_update",
    "qef_assignment",
    "band",
    "tol",
}


def dual_contouring(grid, level=0.0, intersection=None, method="ju", **options):
    """Extract an iso-surface with dual contouring.

    Every cell the surface crosses gets one vertex and every crossed grid
    edge one quad connecting the four cells around it. The vertex placement
    depends on the variant:

    ``"ju"`` (default) is dual contouring of Hermite data (Ju et al. 2002):
    each vertex minimizes the QEF of the tangent planes at the cell's edge
    crossings. It needs normals; estimated from the grid values they are
    smeared at sharp features, exact SDF normals attached to the
    intersection reproduce them. One linear solve per cell.

    ``"carrera"`` is dual contouring of signed distance data (Carrera et
    al. 2026). The vertices are optimized so that the mesh is tangent to
    the spheres around the grid samples near the surface, of radius equal to
    the absolute sample value,
    which recovers sharp features from the samples alone, without normals.
    The grid must hold a signed distance field. Iterative, and orders of
    magnitude slower than ``"ju"``.

    Args:
        grid: The input grid containing scalar values.
        level: The iso-value. Default is 0.0.
        intersection: Optional Intersection from get_intersection(). Both
            variants use its points as the edge crossings; normals attached
            to it are used by "ju" directly and by "carrera" as the initial
            Hermite normals. Computed automatically when omitted.
        method: "ju" (default) or "carrera".
        reg: (ju) Regularization weight of the QEF. Default 0.01.
        svd_tol: (ju) Relative eigenvalue cutoff of the QEF solve.
            Default 1e-6.
        clamp: (ju) Keep each vertex inside its cell. Default True.
        outer_iters: (carrera) Global iterations. Default 100.
        inner_iters: (carrera) Per-cell iterations per outer iteration.
            Default 100.
        mu: (carrera) Regularization toward the previous iterate.
            Default 0.1.
        hermite_weight: (carrera) Weight of the Hermite plane terms.
            Default 0.02.
        update_weight: (carrera) Blend weight of the per-iteration Hermite
            and face point updates. Default 0.2.
        hermite_update: (carrera) Refine the Hermite data from the mesh
            each iteration. Default True.
        qef_assignment: (carrera) Assign the samples of the first iteration
            on the mesh of QEF vertices instead of the centroid mesh, as the
            reference code does. Default True.
        band: (carrera) Only samples within this many cell diagonals of the
            surface are used. Default 3.0.
        tol: (carrera) Inner-loop stopping step, in cell diagonals.
            Default 1e-5.

    Returns:
        A tuple (vertices, faces) where vertices is an (N, 3) float32
        tensor and faces is an (M, 3) int32 tensor of triangle indices.
    """
    if method == "carrera":
        fn, allowed = _dual_contouring_sdf, _CARRERA_OPTIONS
    elif method == "ju":
        fn, allowed = _dual_contouring_ju, _JU_OPTIONS
    else:
        raise ValueError(f"Unknown dual contouring method {method!r}; expected 'ju' or 'carrera'")
    unknown = set(options) - allowed
    if unknown:
        raise TypeError(f"Options {sorted(unknown)} do not apply to method {method!r}; allowed: {sorted(allowed)}")
    return fn(grid, level, intersection, **options)
