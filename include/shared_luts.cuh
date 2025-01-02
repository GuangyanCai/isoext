#pragma once

// Our cell is defined as follows:
//        v3------e10-----v7
//       /|               /|
//      / |              / |
//    e1  |            e5  |
//    /  e2            /   e6
//   /    |           /    |
//  v1------e9------v5     |
//  |     |          |     |
//  |    v2------e11-|----v6
//  |    /           |    /
// e0  e3           e4  e7
//  |  /             |  /
//  | /              | /
//  |/               |/
//  v0------e8------v4
//
//  z
//  |  y
//  | /
//  |/
//  +----x
//
// This ASCII art represents a 3D cube with:
// - Vertices labeled v0 to v7 in Morton order
// - Edges labeled e0 to e11
// - Front, top, and right faces visible
//
// Vertex mapping in Morton order:
// v0: (0,0,0)  v1: (0,0,1)  v2: (0,1,0)  v3: (0,1,1)
// v4: (1,0,0)  v5: (1,0,1)  v6: (1,1,0)  v7: (1,1,1)
//
// Edge mapping:
// e0: v0-v1   e1: v1-v3   e2: v2-v3   e3: v0-v2
// e4: v4-v5   e5: v5-v7   e6: v6-v7   e7: v4-v6
// e8: v0-v4   e9: v1-v5   e10: v3-v7  e11: v2-v6

// LUTs for cells.

// LUT for edges. Every two elements define an edge.
// The first element is always the smaller index.
const size_t edges_size = 24;
extern const int edges_table[edges_size];

// LUT for edge intersection status.
const size_t edge_table_size = 256;
extern const int edge_status_table[edge_table_size];

// LUT for neighbor cells of a given point.
// Give a point in uint3, its ith neighbor is given by subtracting
// point_neighbor_table[i] from the point.
const size_t point_neighbors_table_size = 8;
extern const uint3 point_neighbors_table[point_neighbors_table_size];

// LUT for neighbor cells of a given edge.
const size_t edge_neighbors_table_size = 12;
extern const uint3 edge_neighbors_table[edge_neighbors_table_size];
