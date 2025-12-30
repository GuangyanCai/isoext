"""Shared test utilities and fixtures."""


def populate_sparse_grid(grid, sdf, shape, level=0.0):
    """Find surface-crossing cells and populate a SparseGrid with SDF values.

    Args:
        grid: A SparseGrid instance.
        sdf: An SDF function that takes points and returns SDF values.
        shape: The grid shape as (x, y, z).
        level: The iso-level to find surface-crossing cells for.
    """
    chunk_size = shape[0] * shape[1] * shape[2]
    chunks = grid.get_potential_cell_indices(chunk_size)

    for chunk in chunks:
        points = grid.get_points_by_cell_indices(chunk)
        sdf_values = sdf(points)
        filtered = grid.filter_cell_indices(chunk, sdf_values, level=level)
        if len(filtered) > 0:
            grid.add_cells(filtered)

    if grid.get_num_cells() > 0:
        points = grid.get_points()
        sdf_values = sdf(points)
        grid.set_values(sdf_values)

