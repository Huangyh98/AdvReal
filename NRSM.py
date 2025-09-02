import numpy as np
import open3d as o3d
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

class PrecomputedTPSDeformer:
    def __init__(self, vertices_np, triangles_np):
        self.vertices = vertices_np
        self.triangles = triangles_np
        self.mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices_np),
            o3d.utility.Vector3iVector(triangles_np)
        )
        self.mesh.compute_vertex_normals()
        self.curvatures = self._compute_curvature()
        self.control_points_coords = None
        self.inverse_kernel = None
        self.target_repr_matrix = None

    def _compute_curvature(self):
        vertices = self.vertices
        triangles = self.triangles
        def calculate_curvature_for_vertex(i):
            adjacent_face_indices = np.where(np.any(triangles == i, axis=1))[0]
            if len(adjacent_face_indices) == 0: return 0.0
            adjacent_faces = triangles[adjacent_face_indices]
            p1, p2, p3 = vertices[adjacent_faces[:, 0]], vertices[adjacent_faces[:, 1]], vertices[adjacent_faces[:, 2]]
            face_normals = np.cross(p2 - p1, p3 - p1)
            norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
            face_normals /= (norm + 1e-8)
            mean_normal = np.mean(face_normals, axis=0)
            mean_normal /= (np.linalg.norm(mean_normal) + 1e-8)
            return np.mean(np.linalg.norm(face_normals - mean_normal, axis=1))
        curvatures = Parallel(n_jobs=-1)(delayed(calculate_curvature_for_vertex)(i) for i in range(len(vertices)))
        return np.nan_to_num(np.array(curvatures), nan=0.0)

    def select_and_prepare(self, num_points=200):
        print(f"🎯 Selecting {num_points} control points based on curvature...")
        sample_weights = self.curvatures / (self.curvatures.sum() + 1e-8)
        num_samples = min(len(self.vertices), num_points * 10)
        candidate_indices = np.random.choice(len(self.vertices), size=num_samples, replace=False, p=sample_weights)
        candidate_points = self.vertices[candidate_indices]
        kmeans = MiniBatchKMeans(n_clusters=num_points, n_init='auto', batch_size=256)
        kmeans.fit(candidate_points)
        control_indices = [np.argmin(np.linalg.norm(self.vertices - center, axis=1)) for center in kmeans.cluster_centers_]
        unique_indices = np.array(list(set(control_indices)))
        self.control_points_coords = self.vertices[unique_indices]
        print(f"✅ Selected {len(self.control_points_coords)} unique control points.")
        print("⚙️ Pre-computing TPS transformation matrices (the 'weights')...")
        self._precompute_tps_matrices()
        print("✅ Pre-computation complete.")

    def _phi(self, r):
        r_safe = np.where(r == 0, 1e-9, r)
        return 0.5 * (r_safe**2) * np.log(r_safe)

    def _precompute_tps_matrices(self):
        C = self.control_points_coords
        V = self.vertices
        N = C.shape[0]
        ndim = C.shape[1]
        pairwise_dist = cdist(C, C)
        kernel_matrix = self._phi(pairwise_dist)
        P = np.hstack([np.ones((N, 1)), C])
        L = np.block([[kernel_matrix, P],[P.T, np.zeros((ndim + 1, ndim + 1))]])
        self.inverse_kernel = np.linalg.inv(L)
        M = V.shape[0]
        pairwise_dist_V_C = cdist(V, C)
        target_kernel_part = self._phi(pairwise_dist_V_C)
        target_poly_part = np.hstack([np.ones((M, 1)), V])
        self.target_repr_matrix = np.hstack([target_kernel_part, target_poly_part])

    def deform(self, displacements_np):
        if self.inverse_kernel is None or self.target_repr_matrix is None:
            raise RuntimeError("Deformer has not been prepared. Call select_and_prepare() first.")
        ndim = self.control_points_coords.shape[1]
        padded_displacements = np.vstack([displacements_np, np.zeros((ndim + 1, ndim))])
        mapping_matrix = self.inverse_kernel @ padded_displacements
        vertex_displacements = self.target_repr_matrix @ mapping_matrix
        return self.vertices + vertex_displacements
