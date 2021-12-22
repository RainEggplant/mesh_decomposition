from collections import deque

import numpy as np
from scipy.sparse import csr_matrix, find
from scipy.sparse.csgraph import johnson

from mesh import Mesh


class BinaryDecomposer:
    def __init__(self, epsilon=0.08, max_iter=100):
        self._eps = 1e-14  # for float comparison
        self._max_flow = float('inf')
        self.epsilon = epsilon
        self.max_iter = max_iter

    @staticmethod
    def calc_all_pair_shortest_paths(graph) -> np.ndarray:
        dist_mat = johnson(csgraph=graph, directed=False)
        return dist_mat

    def solve_representing_patches(self, dist_mat):
        # choose points with max distance as initial representing patches REP_A and REP_B
        rep_a, rep_b = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        prob_a = dist_mat[rep_a] / (dist_mat[rep_a] + dist_mat[rep_b])
        prob_b = 1 - prob_a

        # iteratively optimize the REPs
        n_verts = dist_mat.shape[0]
        for i in range(self.max_iter):
            energy_min = np.dot(prob_a, dist_mat[rep_a])
            rep_a_new = rep_a
            for rep in range(n_verts):
                energy = np.dot(prob_a, dist_mat[rep])
                if energy < energy_min:
                    energy_min = energy
                    rep_a_new = rep

            energy_min = np.dot(prob_b, dist_mat[rep_b])
            rep_b_new = rep_b
            for rep in range(n_verts):
                energy = np.dot(prob_b, dist_mat[rep])
                if energy < energy_min:
                    energy_min = energy
                    rep_b_new = rep

            if (rep_a_new == rep_a and rep_b_new == rep_b) or (rep_a_new == rep_b and rep_b_new == rep_a):
                # it also converges when rep_a and rep_b are swapping
                break

            rep_a = rep_a_new
            rep_b = rep_b_new
            prob_a = dist_mat[rep_a] / (dist_mat[rep_a] + dist_mat[rep_b])
            prob_b = 1 - prob_a

        return rep_a, rep_b

    def generate_fuzzy_decomposition(self, dist_mat, rep_a, rep_b):
        prob_a = dist_mat[rep_a] / (dist_mat[rep_a] + dist_mat[rep_b])

        # decompose the dual mesh into 3 patches
        a_verts = np.argwhere(prob_a > 0.5 + self.epsilon).flatten()
        b_verts = np.argwhere(prob_a < 0.5 - self.epsilon).flatten()

        # use average distance to refine as stated in the paper
        for i in range(self.max_iter):
            prob_a_prev = prob_a
            a_f = np.mean(dist_mat[a_verts], axis=0)
            b_f = np.mean(dist_mat[b_verts], axis=0)
            prob_a = a_f / (a_f + b_f)
            a_verts = np.argwhere(prob_a > 0.5 + self.epsilon).flatten()
            b_verts = np.argwhere(prob_a < 0.5 - self.epsilon).flatten()
            if np.all((prob_a - prob_a_prev) < self._eps) or np.all((prob_a + prob_a_prev - 1) < self._eps):
                break

        fuzzy_verts = np.argwhere((0.5 - self.epsilon <= prob_a) & (prob_a <= 0.5 + self.epsilon)).flatten()
        return a_verts, b_verts, fuzzy_verts

    def construct_duzzy_graph(self, n_verts, face_adj, ang_dists, avg_ang_dist, a_verts, fuzzy_verts):
        row = []
        col = []
        data = []
        v_s = n_verts  # flow source point
        v_t = n_verts + 1  # flow target point

        for pair_id, (p, q) in enumerate(face_adj):
            p_uncertain = p in fuzzy_verts
            p_in_a = p in a_verts
            p_in_b = not (p_uncertain and p_in_a)

            q_uncertain = q in fuzzy_verts
            q_in_a = q in a_verts

            if (not p_uncertain) and (not q_uncertain):
                continue

            weight = 1 / (1 + ang_dists[pair_id] / avg_ang_dist)
            row.extend([p, q])
            col.extend([q, p])
            data.extend([weight, weight])
            if not (p_uncertain and q_uncertain):
                if p_in_a or q_in_a:
                    v_a = p if p_in_a else q
                    row.extend([v_s, v_a])
                    col.extend([v_a, v_s])
                    data.extend([self._max_flow, 0])
                else:
                    v_b = p if p_in_b else q
                    row.extend([v_b, v_t])
                    col.extend([v_t, v_b])
                    data.extend([self._max_flow, 0])

        fuzzy_graph = csr_matrix((data, (row, col)), shape=(n_verts + 2, n_verts + 2))
        return fuzzy_graph

    # Ford–Fulkerson Algorithm
    # ref to "https://www.geeksforgeeks.org/minimum-cut-in-a-directed-graph"
    @staticmethod
    def _bfs(graph, v_s, v_t, parent):
        visited = [False] * graph.shape[0]
        queue = deque()

        queue.append(v_s)
        visited[v_s] = True
        while queue:
            v = queue.popleft()
            for v_adj in find(graph[v])[1]:
                if not visited[v_adj]:
                    visited[v_adj] = True
                    queue.append(v_adj)
                    parent[v_adj] = v
                    if v_adj == v_t:
                        return True

        # did not reach the flow target point starting from source
        return False

    @staticmethod
    def _dfs(graph, v, visited):
        visited[v] = True
        for v_adj in find(graph[v])[1]:
            if not visited[v_adj]:
                BinaryDecomposer._dfs(graph, v_adj, visited)

    def minimum_cut(self, graph, v_s, v_t):
        cut_graph = graph.copy()
        parent = [-1] * cut_graph.shape[0]  # store the path filled by BFS

        # augment the flow while there is path from source to target
        while self._bfs(cut_graph, v_s, v_t, parent):
            # find the minimum residual capacity of the edges along the path filled by BFS.
            # (or find the maximum flow through the path found, equivalently).
            path_flow = self._max_flow
            v = v_t
            while (v != v_s):
                path_flow = min(path_flow, cut_graph[parent[v], v])
                v = parent[v]

            # update residual capacities of the edges and reverse edges along the path
            v = v_t
            while (v != v_s):
                u = parent[v]
                cut_graph[u, v] -= path_flow
                if cut_graph[u, v] < self._eps:  # for numerical stability
                    cut_graph[u, v] = 0
                cut_graph[v, u] += path_flow
                if cut_graph[v, u] < self._eps:
                    cut_graph[v, u] = 0
                v = parent[v]

        visited = [False] * cut_graph.shape[0]
        self._dfs(cut_graph, v_s, visited)

        src_side_verts = np.argwhere(visited).flatten()
        return src_side_verts

    def decompose(self, mesh: Mesh):
        dist_mat = self.calc_all_pair_shortest_paths(mesh.dual_graph)
        rep_a, rep_b = self.solve_representing_patches(dist_mat)
        a_verts, b_verts, fuzzy_verts = self.generate_fuzzy_decomposition(dist_mat, rep_a, rep_b)
        fuzzy_graph = self.construct_duzzy_graph(mesh.faces.shape[0], mesh.face_adj, mesh.ang_dists, mesh.avg_ang_dist,
                                                 a_verts, fuzzy_verts)

        v_s = dist_mat.shape[0]
        v_t = dist_mat.shape[0] + 1
        src_side_verts = self.minimum_cut(fuzzy_graph, v_s, v_t)

        a_fuzzy_verts = set(fuzzy_verts) & set(src_side_verts)
        b_fuzzy_verts = set(fuzzy_verts) - set(a_verts)
        a_fuzzy_verts = np.array(list(a_fuzzy_verts))
        b_fuzzy_verts = np.array(list(b_fuzzy_verts))
        return a_verts, a_fuzzy_verts, b_verts, b_fuzzy_verts
