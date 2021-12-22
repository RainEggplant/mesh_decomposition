import argparse
import colorsys
from pathlib import Path

import numpy as np
import trimesh

from decomposer import BinaryDecomposer
from mesh import Mesh


def get_color(n_colors):
    colors = []
    for i in np.arange(0, 360, 360 / n_colors):
        hue = i / 360
        lightness = (50 + np.random.rand() * 10) / 100
        saturation = (90 + np.random.rand() * 10) / 100
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def main(args):
    mesh = Mesh(args.input)
    mesh.construct_dual_graph(args.eta, args.delta)
    if args.method == 'binary':
        decomposer = BinaryDecomposer(args.epsilon)
        a_verts, a_fuzzy_verts, b_verts, b_fuzzy_verts = decomposer.decompose(mesh)

        # assign face colors
        colors = get_color(2)
        n_verts = mesh.faces.shape[0]
        face_colors = np.zeros((n_verts, 3))
        face_colors[a_verts] = colors[0]
        face_colors[a_fuzzy_verts] = colors[0]
        face_colors[b_verts] = colors[1]
        face_colors[b_fuzzy_verts] = colors[1]
    else:
        raise NotImplementedError(f'method [{args.method}] has not been implemented!')

    mesh_colored = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces, face_colors=face_colors)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mesh_colored.export(args.output)
    if args.show:
        mesh_colored.show(resolution=(600, 600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input mesh file')
    parser.add_argument('-o', '--output', type=str, required=True, help='output decomposed mesh file')
    parser.add_argument('-m', '--method', choices=['binary', 'k-way'], default='binary', help='decomposition method')
    parser.add_argument('--show', action='store_true', help='whether to show the colored decomposed mesh '
                                                            '(requires `pyglet`)')
    parser.add_argument('--eta', type=float, default=0.2, help='weight for convex angular distance')
    parser.add_argument('--delta', type=float, default=0.5, help='weight for geodesic distance')
    parser.add_argument('--epsilon', type=float, default=0.08, help='half interval for fuzzy region')

    args = parser.parse_args()
    main(args)
