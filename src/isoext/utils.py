
def write_obj(obj_path, v, f):
    with open(obj_path, 'w') as obj_file:
        v = v.tolist()
        f = (f + 1).tolist()

        lines = []
        for v0, v1, v2 in v:
            lines.append(f'v {v0} {v1} {v2}\n')

        for f0, f1, f2 in f:
            lines.append(f'f {f0} {f1} {f2}\n')

        obj_file.writelines(lines)