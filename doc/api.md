# API Reference

## Functions

```{eval-rst}
.. autofunction:: isoext.marching_cubes

.. autofunction:: isoext.marching_tetrahedra

.. autofunction:: isoext.dual_contouring

.. autofunction:: isoext.surface_nets

.. autofunction:: isoext.get_intersection

.. autofunction:: isoext.make_grid

.. autofunction:: isoext.gaussian_smooth

.. autofunction:: isoext.write_obj
```

(api-viewer)=
## Viewer

Interactive visualization built on [viser](https://viser.studio).

```{eval-rst}
.. automodule:: isoext.viewer
   :members: show, embed, add_mesh, add_grid, add_points, add_lines, add_arrows, add_planes, serialize_scene, save_scene, copy_client
```

## Classes

### UniformGrid

```{eval-rst}
.. py:class:: isoext.UniformGrid(shape, aabb_min=[-1,-1,-1], aabb_max=[1,1,1], default_value=float_max)

   A dense uniform grid for storing scalar values.

   The grid divides a 3D axis-aligned bounding box into a regular lattice of cells.
   Each cell has 8 corner points where scalar values are stored.

   :param shape: The number of sample points in each dimension (x, y, z);
      the grid has one fewer cell than points along each axis.
   :type shape: Sequence[int]
   :param aabb_min: The minimum corner of the bounding box.
   :type aabb_min: Sequence[float]
   :param aabb_max: The maximum corner of the bounding box.
   :type aabb_max: Sequence[float]
   :param default_value: Initial scalar value for all points.
   :type default_value: float

   .. automethod:: get_points
   .. automethod:: get_values
   .. automethod:: set_values
```

### SparseGrid

```{eval-rst}
.. py:class:: isoext.SparseGrid(shape, aabb_min=[-1,-1,-1], aabb_max=[1,1,1], default_value=float_max)

   A sparse adaptive grid for storing scalar values.

   Unlike UniformGrid, SparseGrid only allocates memory for cells that are explicitly added.
   This is useful for large domains where only a small region contains the iso-surface.

   :param shape: The number of sample points in each dimension (x, y, z);
      cells may be added anywhere in the implied lattice.
   :type shape: Sequence[int]
   :param aabb_min: The minimum corner of the bounding box.
   :type aabb_min: Sequence[float]
   :param aabb_max: The maximum corner of the bounding box.
   :type aabb_max: Sequence[float]
   :param default_value: Default scalar value for unset points.
   :type default_value: float

   .. automethod:: get_num_cells
   .. automethod:: get_num_points
   .. automethod:: get_points
   .. automethod:: get_values
   .. automethod:: set_values
   .. automethod:: get_cells
   .. automethod:: add_cells
   .. automethod:: remove_cells
   .. automethod:: get_cell_indices
   .. automethod:: get_potential_cell_indices
   .. automethod:: get_points_by_cell_indices
   .. automethod:: filter_cell_indices
```

### Intersection

```{eval-rst}
.. autoclass:: isoext.Intersection
   :members:
   :undoc-members:
```

