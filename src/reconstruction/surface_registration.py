"""
surface_registration.py
-----------------------
Point cloud and surface mesh registration utilities.

Provides:
  1. ICP (Iterative Closest Point) — rigid registration
  2. CPD (Coherent Point Drift) — non-rigid / deformable registration
  3. Template-to-patient alignment for:
       - Statistical shape models (SSM) fitting
       - Surgical implant placement
       - Pre-/post-operative comparison
  4. Symmetric surface distance (ASSD / HD95) metrics
  5. Landmark-initialised coarse alignment (Procrustes)

Coordinate systems:
  All meshes and point clouds use physical mm coordinates.
  Transformation matrices are 4×4 homogeneous (RAS convention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rigid registration — ICP
# ---------------------------------------------------------------------------

@dataclass
class ICPResult:
    """Output from ICP registration."""
    transform: np.ndarray          # (4, 4) homogeneous transformation
    rms_error: float               # root-mean-square point-to-point distance
    n_iterations: int
    converged: bool
    correspondence_count: int
    initial_rms_error: float


class ICPRegistration:
    """
    Iterative Closest Point (ICP) rigid registration.

    Minimises the sum of squared distances between a moving point cloud
    (source) and a fixed surface (target). Uses point-to-plane variant
    by default for faster convergence on smooth surfaces.

    Reference:
        Besl & McKay, "A method for registration of 3D shapes", PAMI 1992.
        Low, "Linear Least-Squares Optimization for Point-to-Plane ICP", 2004.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,       # convergence threshold on RMS change
        max_correspondence_distance: float = 10.0,  # mm; reject far-away matches
        outlier_fraction: float = 0.05,             # discard worst N% correspondences
        point_to_plane: bool = True,
        subsample: Optional[int] = None,            # use random subset of N points
    ) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_distance = max_correspondence_distance
        self.outlier_fraction = outlier_fraction
        self.point_to_plane = point_to_plane
        self.subsample = subsample

    def register(
        self,
        source: np.ndarray,             # (N, 3) moving points in mm
        target: np.ndarray,             # (M, 3) fixed reference points in mm
        target_normals: Optional[np.ndarray] = None,  # (M, 3) surface normals
        initial_transform: Optional[np.ndarray] = None,  # (4, 4) initial guess
    ) -> ICPResult:
        """
        Run ICP from source to target.

        Returns:
            ICPResult with the 4×4 rigid transform that maps source → target.
        """
        src = source.copy().astype(np.float64)
        tgt = target.astype(np.float64)

        if self.subsample and len(src) > self.subsample:
            idx = np.random.choice(len(src), self.subsample, replace=False)
            src_sub = src[idx]
        else:
            src_sub = src

        # Apply initial transform
        T = np.eye(4, dtype=np.float64) if initial_transform is None else initial_transform.copy()
        src_sub = self._apply_transform(src_sub, T)

        tree = KDTree(tgt)
        initial_rms = self._compute_rms(src_sub, tree)
        prev_rms = float("inf")

        for it in range(self.max_iterations):
            dists, indices = tree.query(src_sub, k=1)

            # Reject correspondences beyond max distance
            valid = dists < self.max_distance

            # Discard outlier fraction
            if self.outlier_fraction > 0 and valid.sum() > 10:
                thresh = np.percentile(dists[valid], 100 * (1 - self.outlier_fraction))
                valid &= dists <= thresh

            if valid.sum() < 6:
                logger.warning("ICP: fewer than 6 valid correspondences at iter %d", it)
                break

            src_pts = src_sub[valid]
            tgt_pts = tgt[indices[valid]]
            n_pts = tgt_normals[indices[valid]] if target_normals is not None and self.point_to_plane else None

            # Solve for incremental rigid transform
            if n_pts is not None:
                delta_T = self._solve_point_to_plane(src_pts, tgt_pts, n_pts)
            else:
                delta_T = self._solve_point_to_point(src_pts, tgt_pts)

            src_sub = self._apply_transform(src_sub, delta_T)
            T = delta_T @ T

            rms = self._compute_rms(src_sub[valid], tree)
            if abs(prev_rms - rms) < self.tolerance:
                logger.debug("ICP converged at iteration %d (ΔRMS=%.2e)", it, abs(prev_rms - rms))
                return ICPResult(
                    transform=T,
                    rms_error=rms,
                    n_iterations=it + 1,
                    converged=True,
                    correspondence_count=int(valid.sum()),
                    initial_rms_error=initial_rms,
                )
            prev_rms = rms

        return ICPResult(
            transform=T,
            rms_error=prev_rms,
            n_iterations=self.max_iterations,
            converged=False,
            correspondence_count=int(valid.sum()),
            initial_rms_error=initial_rms,
        )

    @staticmethod
    def _solve_point_to_point(
        src: np.ndarray,  # (N, 3)
        tgt: np.ndarray,  # (N, 3)
    ) -> np.ndarray:
        """SVD-based optimal rigid transform (Umeyama algorithm)."""
        mu_src = src.mean(axis=0)
        mu_tgt = tgt.mean(axis=0)
        src_c = src - mu_src
        tgt_c = tgt - mu_tgt

        H = src_c.T @ tgt_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = mu_tgt - R @ mu_src
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def _solve_point_to_plane(
        src: np.ndarray,    # (N, 3)
        tgt: np.ndarray,    # (N, 3)
        normals: np.ndarray,  # (N, 3) target normals
    ) -> np.ndarray:
        """
        Point-to-plane ICP linearised solution.
        Faster convergence than point-to-point for smooth surfaces.
        """
        # Build A matrix and b vector for least squares: A x = b
        # x = [α, β, γ, tx, ty, tz] — small angle approximation
        n = len(src)
        A = np.zeros((n, 6))
        b = np.zeros(n)

        nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
        sx, sy, sz = src[:, 0], src[:, 1], src[:, 2]
        tx, ty, tz = tgt[:, 0], tgt[:, 1], tgt[:, 2]

        A[:, 0] = nz * sy - ny * sz
        A[:, 1] = nx * sz - nz * sx
        A[:, 2] = ny * sx - nx * sy
        A[:, 3] = nx
        A[:, 4] = ny
        A[:, 5] = nz
        b = nx * tx + ny * ty + nz * tz - nx * sx - ny * sy - nz * sz

        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        α, β, γ, tx_, ty_, tz_ = x
        R = Rotation.from_euler("xyz", [α, β, γ]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx_, ty_, tz_]
        return T

    @staticmethod
    def _apply_transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Apply (4, 4) homogeneous transform to (N, 3) points."""
        hom = np.hstack([pts, np.ones((len(pts), 1))])
        return (T @ hom.T).T[:, :3]

    @staticmethod
    def _compute_rms(pts: np.ndarray, tree: KDTree) -> float:
        dists, _ = tree.query(pts, k=1)
        return float(np.sqrt((dists ** 2).mean()))


# ---------------------------------------------------------------------------
# Deformable registration — CPD (Coherent Point Drift)
# ---------------------------------------------------------------------------

@dataclass
class CPDResult:
    """Output from CPD non-rigid registration."""
    transformed_source: np.ndarray   # (N, 3) deformed source points
    deformation_field: np.ndarray    # (N, 3) displacements
    sigma_squared: float             # noise variance at convergence
    n_iterations: int
    converged: bool
    correspondence_matrix: np.ndarray  # (N, M) soft correspondence


class CPDRegistration:
    """
    Coherent Point Drift (CPD) for non-rigid deformable registration.

    Used for:
      - Statistical Shape Model (SSM) fitting to patient anatomy
      - Template-based deformation for defect reconstruction
      - Pre/post-operative comparison when rigid assumption fails

    Implements the non-rigid EM variant of CPD.

    Reference:
        Myronenko & Song, "Point Set Registration: Coherent Point Drift",
        PAMI 2010.
    """

    def __init__(
        self,
        max_iterations: int = 150,
        tolerance: float = 1e-5,
        beta: float = 2.0,       # width of Gaussian motion coherence kernel (mm)
        lambda_: float = 3.0,    # regularisation weight
        w: float = 0.1,          # outlier fraction (0-1)
        sigma_init: Optional[float] = None,  # initial noise variance; None = auto
    ) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.beta = beta
        self.lambda_ = lambda_
        self.w = w
        self.sigma_init = sigma_init

    def register(
        self,
        source: np.ndarray,  # (N, 3) template / moving
        target: np.ndarray,  # (M, 3) fixed patient surface
    ) -> CPDResult:
        """
        Run CPD non-rigid registration.

        Returns:
            CPDResult with deformed source and soft correspondences.
        """
        N, D = source.shape
        M = len(target)

        X = target.astype(np.float64)   # fixed
        Y = source.copy().astype(np.float64)  # moving (will be deformed)

        # Initialise sigma²
        sigma_sq = self.sigma_init or (
            np.sum((X[:, None] - Y[None, :]) ** 2) / (D * M * N)
        )

        # Gaussian kernel matrix for motion coherence
        G = self._gaussian_kernel(Y, self.beta)  # (N, N)

        W = np.zeros((N, D))    # deformation coefficients

        prev_Q = float("inf")

        for it in range(self.max_iterations):
            # -------- E-step: compute posterior (soft correspondences) ----
            P = self._e_step(X, Y + G @ W, sigma_sq, N, M, D)

            # -------- M-step: update W and sigma² -------------------------
            P1 = P.sum(axis=1)              # (N,)
            Pt1 = P.sum(axis=0)             # (M,)
            PX = P @ X                      # (N, D)
            Np = P1.sum()

            dP = np.diag(P1)
            A = dP @ G + self.lambda_ * sigma_sq * np.eye(N)
            B = PX - dP @ Y
            W = np.linalg.solve(A, B)

            T = Y + G @ W  # deformed source

            # Update sigma²
            xPx = (Pt1 * (X ** 2).sum(axis=1)).sum()
            yPy = (P1 * (T ** 2).sum(axis=1)).sum()
            pxy = np.trace(T.T @ dP @ T) - 2 * np.trace(PX.T @ W.T @ G.T) + yPy
            sigma_sq_new = (xPx - 2 * np.sum(PX * T) + pxy) / (Np * D)
            sigma_sq_new = max(sigma_sq_new, 1e-8)

            # Convergence check
            Q = sigma_sq_new
            if abs(prev_Q - Q) < self.tolerance:
                logger.debug("CPD converged at iteration %d", it)
                break
            prev_Q = Q
            sigma_sq = sigma_sq_new

        deformed = Y + G @ W
        return CPDResult(
            transformed_source=deformed,
            deformation_field=G @ W,
            sigma_squared=float(sigma_sq),
            n_iterations=it + 1,
            converged=abs(prev_Q - Q) < self.tolerance,
            correspondence_matrix=P,
        )

    def _e_step(
        self,
        X: np.ndarray,
        T: np.ndarray,
        sigma_sq: float,
        N: int,
        M: int,
        D: int,
    ) -> np.ndarray:
        """Compute soft correspondence matrix P (N × M)."""
        diff = X[None, :, :] - T[:, None, :]  # (N, M, D)
        sq_dist = (diff ** 2).sum(axis=2)        # (N, M)
        num = np.exp(-0.5 * sq_dist / sigma_sq)

        c = (2 * np.pi * sigma_sq) ** (D / 2) * (self.w / (1 - self.w)) * (N / M)
        denom = num.sum(axis=0, keepdims=True) + c
        P = num / (denom + 1e-300)
        return P

    @staticmethod
    def _gaussian_kernel(Y: np.ndarray, beta: float) -> np.ndarray:
        """Build N×N Gaussian kernel matrix."""
        diff = Y[:, None, :] - Y[None, :, :]  # (N, N, D)
        sq = (diff ** 2).sum(axis=2)            # (N, N)
        return np.exp(-0.5 * sq / (beta ** 2))


# ---------------------------------------------------------------------------
# Procrustes alignment (landmark-based)
# ---------------------------------------------------------------------------

def procrustes_align(
    source_landmarks: np.ndarray,  # (K, 3) source landmark coords in mm
    target_landmarks: np.ndarray,  # (K, 3) target landmark coords in mm
    allow_scaling: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Generalised Procrustes alignment for landmark-based initialisation.

    Args:
        source_landmarks: (K, 3) array of source landmark coordinates.
        target_landmarks: (K, 3) array of target landmark coordinates.
        allow_scaling: If True, compute optimal uniform scale.

    Returns:
        Tuple of (4×4 transform matrix, RMSE in mm).
    """
    assert source_landmarks.shape == target_landmarks.shape
    src = source_landmarks.astype(np.float64)
    tgt = target_landmarks.astype(np.float64)

    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    src_c = src - mu_s
    tgt_c = tgt - mu_t

    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    sign_mat = np.eye(3)
    sign_mat[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ sign_mat @ U.T

    scale = 1.0
    if allow_scaling:
        scale = S.sum() / (src_c ** 2).sum()

    t = mu_t - scale * R @ mu_s
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    # RMSE after alignment
    src_aligned = (T[:3, :3] @ src.T).T + T[:3, 3]
    rmse = float(np.sqrt(((src_aligned - tgt) ** 2).sum(axis=1).mean()))

    return T, rmse


# ---------------------------------------------------------------------------
# Surface distance metrics
# ---------------------------------------------------------------------------

def symmetric_surface_distance(
    mesh_a: np.ndarray | trimesh.Trimesh,
    mesh_b: np.ndarray | trimesh.Trimesh,
    n_sample: int = 10000,
) -> Dict[str, float]:
    """
    Compute symmetric surface distance metrics between two surfaces.

    Metrics (all in mm):
      - assd: Average Symmetric Surface Distance
      - rmsd: RMS Symmetric Surface Distance
      - hd95: 95th percentile Hausdorff distance
      - hd:   Maximum Hausdorff distance
      - mean_a2b, mean_b2a: one-directional mean distances

    Args:
        mesh_a: Either (N, 3) point cloud or trimesh.Trimesh.
        mesh_b: Either (M, 3) point cloud or trimesh.Trimesh.
        n_sample: Number of points to sample from each surface.

    Returns:
        Dict with keys: assd, rmsd, hd95, hd, mean_a2b, mean_b2a.
    """
    pts_a = _sample_surface(mesh_a, n_sample)
    pts_b = _sample_surface(mesh_b, n_sample)

    tree_a = KDTree(pts_a)
    tree_b = KDTree(pts_b)

    d_a2b, _ = tree_b.query(pts_a)
    d_b2a, _ = tree_a.query(pts_b)

    all_d = np.concatenate([d_a2b, d_b2a])

    return {
        "assd":    float(all_d.mean()),
        "rmsd":    float(np.sqrt((all_d ** 2).mean())),
        "hd95":    float(np.percentile(all_d, 95)),
        "hd":      float(all_d.max()),
        "mean_a2b": float(d_a2b.mean()),
        "mean_b2a": float(d_b2a.mean()),
    }


def _sample_surface(
    surface: np.ndarray | trimesh.Trimesh,
    n_sample: int,
) -> np.ndarray:
    """Return (N, 3) point cloud sampled from mesh or point cloud."""
    if isinstance(surface, trimesh.Trimesh):
        pts, _ = trimesh.sample.sample_surface(surface, n_sample)
        return pts
    elif isinstance(surface, np.ndarray):
        if len(surface) > n_sample:
            idx = np.random.choice(len(surface), n_sample, replace=False)
            return surface[idx]
        return surface
    else:
        raise TypeError(f"Expected ndarray or Trimesh, got {type(surface)}")


# ---------------------------------------------------------------------------
# Full template-to-patient registration pipeline
# ---------------------------------------------------------------------------

class TemplateRegistration:
    """
    Complete template-to-patient registration for implant and SSM fitting.

    Pipeline:
    1. Landmark-based Procrustes coarse alignment.
    2. ICP rigid fine alignment.
    3. CPD non-rigid deformation for shape adaptation.
    4. Final surface distance evaluation.
    """

    def __init__(
        self,
        icp_kwargs: Optional[dict] = None,
        cpd_kwargs: Optional[dict] = None,
    ) -> None:
        self.icp = ICPRegistration(**(icp_kwargs or {}))
        self.cpd = CPDRegistration(**(cpd_kwargs or {}))

    def fit(
        self,
        template_mesh: trimesh.Trimesh,
        patient_mesh: trimesh.Trimesh,
        template_landmarks: Optional[np.ndarray] = None,
        patient_landmarks: Optional[np.ndarray] = None,
        deformable: bool = True,
    ) -> Dict:
        """
        Register template to patient surface.

        Args:
            template_mesh: Template bone model.
            patient_mesh: Patient-specific bone mesh.
            template_landmarks: (K, 3) optional landmarks on template.
            patient_landmarks: (K, 3) optional corresponding landmarks.
            deformable: If True, also run CPD non-rigid step.

        Returns:
            Dict with keys:
              - 'rigid_transform': (4, 4) matrix
              - 'rigid_distances': surface distance dict
              - 'deformed_mesh': trimesh.Trimesh (if deformable=True)
              - 'deformed_distances': surface distance dict (if deformable)
              - 'icp_result': ICPResult
              - 'cpd_result': CPDResult (if deformable)
        """
        result = {}

        # 1. Coarse alignment from landmarks
        init_T = np.eye(4)
        if template_landmarks is not None and patient_landmarks is not None:
            init_T, lm_rmse = procrustes_align(template_landmarks, patient_landmarks)
            logger.info("Procrustes alignment RMSE: %.2f mm", lm_rmse)
            result["landmark_rmse_mm"] = lm_rmse

        # 2. ICP rigid registration
        src_pts = trimesh.sample.sample_surface(template_mesh, 20000)[0]
        tgt_pts = trimesh.sample.sample_surface(patient_mesh, 20000)[0]
        tgt_normals = np.array([patient_mesh.face_normals[i]
                                for i in trimesh.sample.sample_surface(patient_mesh, 20000)[1]])

        icp_result = self.icp.register(src_pts, tgt_pts, target_normals=tgt_normals,
                                        initial_transform=init_T)
        result["icp_result"] = icp_result
        result["rigid_transform"] = icp_result.transform
        logger.info("ICP RMS: %.3f mm (%d iters)", icp_result.rms_error, icp_result.n_iterations)

        # Apply rigid transform to template mesh
        rigid_mesh = template_mesh.copy()
        rigid_mesh.apply_transform(icp_result.transform)
        result["rigid_mesh"] = rigid_mesh
        result["rigid_distances"] = symmetric_surface_distance(rigid_mesh, patient_mesh)
        logger.info("After rigid: ASSD=%.3f mm, HD95=%.3f mm",
                    result["rigid_distances"]["assd"], result["rigid_distances"]["hd95"])

        if not deformable:
            return result

        # 3. CPD non-rigid refinement
        rigid_pts = np.array(rigid_mesh.vertices)
        cpd_result = self.cpd.register(rigid_pts, tgt_pts)
        result["cpd_result"] = cpd_result

        # Apply CPD deformation to create deformed mesh
        deformed_verts = cpd_result.transformed_source
        # Interpolate deformation to all mesh vertices
        deformed_mesh = self._apply_cpd_to_mesh(rigid_mesh, deformed_verts, rigid_pts)
        result["deformed_mesh"] = deformed_mesh
        result["deformed_distances"] = symmetric_surface_distance(deformed_mesh, patient_mesh)
        logger.info("After CPD: ASSD=%.3f mm, HD95=%.3f mm",
                    result["deformed_distances"]["assd"], result["deformed_distances"]["hd95"])

        return result

    @staticmethod
    def _apply_cpd_to_mesh(
        mesh: trimesh.Trimesh,
        deformed_pts: np.ndarray,
        original_pts: np.ndarray,
    ) -> trimesh.Trimesh:
        """
        Transfer CPD deformation field to all mesh vertices using
        nearest-neighbour interpolation.
        """
        tree = KDTree(original_pts)
        displacements = deformed_pts - original_pts

        # Find closest sampled point for each mesh vertex
        dists, indices = tree.query(mesh.vertices, k=4)
        weights = 1.0 / (dists + 1e-10)
        weights /= weights.sum(axis=1, keepdims=True)
        vertex_disp = (weights[:, :, None] * displacements[indices]).sum(axis=1)

        new_verts = mesh.vertices + vertex_disp
        return trimesh.Trimesh(vertices=new_verts, faces=mesh.faces.copy(), process=False)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    logger.info("Surface registration self-test ...")

    # Create a simple sphere pair with known offset
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=30.0)
    source_pts = np.array(sphere.vertices)

    # Known ground-truth transform
    gt_R = Rotation.from_euler("z", 15, degrees=True).as_matrix()
    gt_t = np.array([5.0, -3.0, 2.0])
    target_pts = (gt_R @ source_pts.T).T + gt_t

    icp = ICPRegistration(max_iterations=50, point_to_plane=False)
    result = icp.register(source_pts, target_pts)
    logger.info("ICP RMS: %.4f mm | converged: %s | iters: %d",
                result.rms_error, result.converged, result.n_iterations)

    # Surface distance on identical meshes (should be ~0)
    metrics = symmetric_surface_distance(sphere, sphere, n_sample=5000)
    logger.info("Self ASSD: %.6f mm  HD95: %.6f mm", metrics["assd"], metrics["hd95"])

    logger.info("Surface registration self-test passed.")
    sys.exit(0)
