import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation as R


class Mesh:
    def __init__(self, filename):
        # 3D mesh
        self._mesh: trimesh.Trimesh = trimesh.load_mesh(filename, process=False, validate=False)
        self.verts = self._mesh.vertices
        self.faces = self._mesh.faces
        self.norms = self._mesh.face_normals

        # adjacent faces
        self.face_adj: np.ndarray = self._mesh.face_adjacency
        self.face_adj_convex = self._mesh.face_adjacency_convex
        self.face_adj_edges = self._mesh.face_adjacency_edges
        self.face_adj_unshared = self._mesh.face_adjacency_unshared

        # weights
        self.ang_dists: np.ndarray
        self.geo_dists: np.ndarray
        self.avg_ang_dist: float
        self.avg_geo_dist: float

        # dual graph
        self.n_dual_verts = self.faces.shape[0]
        self.ang_weights: csr_matrix
        self.geo_weights: csr_matrix
        self.dual_graph: csr_matrix

    def construct_dual_graph(self, eta_convex=0.2, delta=0.5):
        ang_dist_list = []
        geo_dist_list = []
        for pair_id, (i, j) in enumerate(self.face_adj):
            ang_dist = self.angular_distance(self.norms[i], self.norms[j], self.face_adj_convex[pair_id], eta_convex)
            ang_dist_list.append(ang_dist)

            geo_dist = self.geodesic_distance(verts_shared=self.verts[self.face_adj_edges[pair_id]],
                                              verts_unshared=self.verts[self.face_adj_unshared[pair_id]])
            geo_dist_list.append(geo_dist)

        self.ang_dists = np.array(ang_dist_list)
        self.geo_dists = np.array(geo_dist_list)
        self.avg_ang_dist = self.ang_dists.mean()
        self.avg_geo_dist = self.geo_dists.mean()

        # construct dual graph
        row = [x[0] for x in self.face_adj]
        col = [x[1] for x in self.face_adj]
        graph_shape = (self.n_dual_verts, self.n_dual_verts)
        self.ang_weights = csr_matrix((self.ang_dists, (row, col)), shape=graph_shape)
        self.geo_weights = csr_matrix((self.geo_dists, (row, col)), shape=graph_shape)
        self.dual_graph = (delta * self.geo_weights / self.avg_geo_dist
                           + (1 - delta) * self.ang_weights / self.avg_ang_dist)

    @staticmethod
    def angular_distance(norm_a, norm_b, is_convex, eta_convex=0.2):
        ang_dist = 1 - np.dot(norm_a, norm_b)
        ang_dist = eta_convex * ang_dist if is_convex else ang_dist
        return ang_dist

    @staticmethod
    def geodesic_distance(verts_shared, verts_unshared):
        # make the rotation axis pass through zero point
        offset = np.array(verts_shared[1])
        verts_shared = np.array(verts_shared) - offset
        verts_unshared = np.array(verts_unshared) - offset

        # get the shared edge
        rot_axis = verts_shared[0] - verts_shared[1]
        rot_axis = rot_axis / np.linalg.norm(rot_axis)

        # calculate the normal of faces
        edge_a = verts_unshared[0] - verts_shared[1]
        edge_b = verts_unshared[1] - verts_shared[1]
        norm_a = np.cross(rot_axis, edge_a)
        norm_a = norm_a / np.linalg.norm(norm_a)
        norm_b = -np.cross(rot_axis, edge_b)
        norm_b = norm_b / np.linalg.norm(norm_b)

        # determine rotation angle
        is_convex = np.dot(norm_a, edge_b) <= 0
        angle = np.arccos(np.dot(norm_a, norm_b))
        angle = angle if is_convex else -angle

        # construct rotation from the axis and angle
        mrp = rot_axis * np.tan(angle / 4)  # Modified Rodrigues Parameter
        rot = R.from_mrp(mrp)

        # calculate the centroids and rotate one to make them coplanar
        centroids = [np.mean([*verts_shared, v], axis=0) for v in verts_unshared]
        centroids[0] = rot.apply(centroids[0])

        geo_dist = np.linalg.norm(centroids[0] - centroids[1])
        return geo_dist

    def show(self, resolution=(600, 600)):
        self._mesh.show(resolution=resolution)
