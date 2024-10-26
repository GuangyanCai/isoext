from pathlib import Path
from argparse import ArgumentParser
import json
from isoext.utils import write_obj
import torch

def gen(case_file: Path, output_path: Path):
    #        v3------e10-----v7
    #       /|               /|
    #      / |              / |
    #    e1  |            e5  |
    #    /  e2            /   e6
    #   /    |           /    |
    #  v1------e9------v5     |
    #  |     |          |     |
    #  |    v2------e11-|----v6
    #  |    /           |    /
    # e0  e3           e4  e7
    #  |  /             |  /
    #  | /              | /
    #  |/               |/
    #  v0------e8------v4
    #
    #  z
    #  |  y
    #  | /
    #  |/
    #  +----x    
    #
    # This ASCII art represents a 3D cube with:
    # - Vertices labeled v0 to v7 in Morton order
    # - Edges labeled e0 to e11
    # - Front, top, and right faces visible
    #
    # Vertex mapping in Morton order:
    # v0: (0,0,0)  v1: (0,0,1)  v2: (0,1,0)  v3: (0,1,1)
    # v4: (1,0,0)  v5: (1,0,1)  v6: (1,1,0)  v7: (1,1,1)
    #
    # Edge mapping:
    # e0: v0-v1   e1: v1-v3   e2: v2-v3   e3: v0-v2
    # e4: v4-v5   e5: v5-v7   e6: v6-v7   e7: v4-v6
    # e8: v0-v4   e9: v1-v5   e10: v3-v7  e11: v2-v6

    num_verts = 8
    num_edges = 12

    verts = [[float(digit) for digit in format(i, "03b")] for i in range(num_verts)]

    # Given the edge index i, the pair of vertices is given by edge_pairs[i:i+2].
    edge_pairs = [
        (0, 1),
        (1, 3),
        (2, 3),
        (0, 2),
        (4, 5),
        (5, 7),
        (6, 7),
        (4, 6),
        (0, 4),
        (1, 5),
        (3, 7),
        (2, 6)
    ]

    # All possible rotations of the cube
    vert_rotations = [
        [0, 1, 2, 3, 4, 5, 6, 7],  # identity
        # Rotate around the spindle going through opposite faces 
        [2, 0, 3, 1, 6, 4, 7, 5],  # rotate 90 degrees around x
        [3, 2, 1, 0, 7, 6, 5, 4],  # rotate 180 degrees around x
        [1, 3, 0, 2, 5, 7, 4, 6],  # rotate 270 degrees around x
        [1, 5, 3, 7, 0, 4, 2, 6],  # rotate 90 degrees around y
        [5, 4, 7, 6, 1, 0, 3, 2],  # rotate 180 degrees around y
        [4, 0, 6, 2, 5, 1, 7, 3],  # rotate 270 degrees around y
        [4, 5, 0, 1, 6, 7, 2, 3],  # rotate 90 degrees around z
        [6, 7, 4, 5, 2, 3, 0, 1],  # rotate 180 degrees around z
        [2, 3, 6, 7, 0, 1, 4, 5],  # rotate 270 degrees around z
        # Rotate around the spindle going through opposite edges
        [1, 0, 5, 4, 3, 2, 7, 6],  # rotate 180 degrees around e0e6
        [7, 3, 5, 1, 6, 2, 4, 0],  # rotate 180 degrees around e1e7
        [7, 6, 3, 2, 5, 4, 1, 0],  # rotate 180 degrees around e2e4
        [2, 6, 0, 4, 3, 7, 1, 5],  # rotate 180 degrees around e3e5
        [4, 6, 5, 7, 0, 2, 1, 3],  # rotate 180 degrees around e8e10
        [7, 5, 6, 4, 3, 1, 2, 0],  # rotate 180 degrees around e9e11
        # Rotate around the spindle going through opposite vertices
        [0, 4, 1, 5, 2, 6, 3, 7],  # rotate 120 degrees around v0v7
        [0, 2, 4, 6, 1, 3, 5, 7],  # rotate 240 degrees around v0v7
        [3, 1, 7, 5, 2, 0, 6, 4],  # rotate 120 degrees around v1v6
        [5, 1, 4, 0, 7, 3, 6, 2],  # rotate 240 degrees around v1v6
        [6, 4, 2, 0, 7, 5, 3, 1],  # rotate 120 degrees around v2v5
        [3, 7, 2, 6, 1, 5, 0, 4],  # rotate 240 degrees around v2v5
        [6, 2, 7, 3, 4, 0, 5, 1],  # rotate 120 degrees around v3v4
        [5, 7, 1, 3, 4, 6, 0, 2],  # rotate 240 degrees around v3v4
    ]
    num_rotations = len(vert_rotations)

    # Generate the edge rotations for each vertex rotation
    edge_rotations = []
    for rotation in vert_rotations:
        edge_rotation = []
        for i, (va, vb) in enumerate(edge_pairs):
            new_va, new_vb = rotation[va], rotation[vb]
            if new_va > new_vb:
                new_va, new_vb = new_vb, new_va
            edge_rotation.append(edge_pairs.index((new_va, new_vb)))
        edge_rotations.append(edge_rotation)

    # Rotate the vertices of a case
    def rotate_verts(case, vert_rotation):
        status = [int(s) for s in case]
        return ''.join(str(status[vert_rotation.index(i)]) for i in range(num_verts))

    # Rotate the triangles of a case
    def rotate_tris(tris, edge_rotation):
        return [[edge_rotation[e] for e in tri] for tri in tris]

    # Flip the vertices of a case
    def reflect_verts(case):
        return ''.join(str(1 - int(s)) for s in case)

    # Flip the triangles of a case
    def reflect_tris(tris):
        return [tri[::-1] for tri in tris]

    # Read the base cases from the case file
    with open(case_file, "r") as f:
        case_file = json.load(f)
    base_cases = case_file["base_cases"]
    use_reflection = case_file.get("use_reflection", True)
    output_path = output_path / case_file["method"]

    # Generate all cases based on the base cases via rotation and reflection
    cases = {}
    for case, tris in base_cases.items():
        for i in range(num_rotations):
            rotated_case = rotate_verts(case, vert_rotations[i])
            if rotated_case not in cases:
                rotated_tris = rotate_tris(tris, edge_rotations[i])
                cases[rotated_case] = rotated_tris
            if use_reflection:
                reflected_case = reflect_verts(rotated_case)
                if reflected_case not in cases:
                    reflected_tris = reflect_tris(rotated_tris)
                    cases[reflected_case] = reflected_tris
    num_cases = len(cases)

    # Check that we have found all 256 cases
    print(f'{num_cases} cases found.')
    if num_cases != 256:
        for i in range(256):
            case = format(i, "08b")
            if case not in cases:
                print(f'case {case} not found')
        exit() 

    # Generate the edge points for the cube
    verts = torch.tensor(verts)
    edge_points = []
    for i in range(num_edges):
        v0, v1 = edge_pairs[i]
        edge_points.append((0.5 * (verts[v0] + verts[v1])))
    edge_points = torch.stack(edge_points)

    # # Output the cases to the test_cases directory
    print(f'Writing meshes for base cases to {output_path / "base_cases"}')
    (output_path / "base_cases").mkdir(exist_ok=True, parents=True)
    for case, tris in base_cases.items():
        faces = torch.tensor(tris)
        write_obj(output_path / "base_cases" / f"{case}.obj", edge_points, faces)

    print(f'Writing meshes for all cases to {output_path / "all_cases"}')
    (output_path / "all_cases").mkdir(exist_ok=True, parents=True)
    for case, tris in cases.items():
        faces = torch.tensor(tris)
        write_obj(output_path / "all_cases" / f"{case}.obj", edge_points, faces)

    # Generate the lookup tables
    print(f'Writing lookup tables to {output_path / "luts"}')
    (output_path / "luts").mkdir(exist_ok=True, parents=True)

    # Generate the edge pairs lookup table
    edge_pairs_lut = torch.tensor(edge_pairs).long().flatten().tolist()
    with open(output_path / "luts" / "edge_pairs.txt", "w") as f:
        for i in range(0, len(edge_pairs_lut), 2):
            f.write(", ".join(str(e) for e in edge_pairs_lut[i:i+2]) + ",\n")

    # Generate the edge status lookup table
    edge_status_lut = []
    for i in range(num_cases):
        case = format(i, "08b")
        tris = cases[case]
        edge_status = [0] * num_edges
        for tri in tris:
            for e in tri:
                edge_status[e] = 1
        edge_status_lut.append("0b" + ''.join(str(e) for e in edge_status[::-1]))

    with open(output_path / "luts" / f"edge_status.txt", "w") as f:
        for i in range(num_cases):
            f.write(edge_status_lut[i] + ",\n")

    # Generate the triangle lookup table
    triangle_lut = []

    # Get the maximum number of triangles for any case
    max_triangles = max(len(tris) for tris in cases.values())
    max_length = max_triangles * 3

    for i in range(num_cases):
        case = format(i, "08b")
        tris = torch.tensor(cases[case]).long().flatten().tolist()
        tris.extend([-1] * (max_length - len(tris)))  # Fill with -1 to max_length
        triangle_lut.extend(tris)

    with open(output_path / "luts" / f"triangle.txt", "w") as f:
        for i in range(0, len(triangle_lut), max_length):
            f.write(", ".join(str(t) for t in triangle_lut[i:i+max_length]) + ",\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("case_file", type=Path)
    parser.add_argument("output_path", type=Path, default=Path("."))
    args = parser.parse_args()
    gen(args.case_file, args.output_path)