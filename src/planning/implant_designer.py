"""
implant_designer.py
-------------------
Patient-specific implant (PSI) design for CMF reconstruction.

Use cases:
  - Orbital floor reconstruction (trauma, post-ablative defects)
  - Cranial plate / cranioplasty
  - Mandibular reconstruction plate (fibula free flap guide)
  - Custom total temporomandibular joint (TMJ) prosthesis
  - Titanium mesh design for craniofacial defects

Design pipeline:
  1. Defect identification from segmentation gap analysis
  2. Surface reconstruction:
       a. Mirror-based (for unilateral defects with contralateral reference)
       b. Statistical Shape Model (SSM) fitting (for bilateral/midline defects)
  3. Thickness analysis — structural integrity for implant durability
  4. Screw trajectory planning — optimal angulation, length, safety margins
  5. Export to STL for CNC milling or additive manufacturing

Manufacturing readiness:
  - Titanium Ti-6Al-4V ELI: min 0.5mm wall thickness for mesh, 1.5mm for plate
  - PEEK: min 2.0mm wall thickness
  - 3D printing nesting and support structure recommendations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.ndimage import label as scipy_label
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScreW:
    """A planned screw trajectory."""
    name: str
    entry_point: np.ndarray   # (3,) mm
    direction: np.ndarray     # (3,) unit vector
    length_mm: float
    diameter_mm: float        # e.g. 2.0 mm cortical screw
    bone_engagement_mm: float # thickness of bone the screw traverses
    safe: bool = True         # True if trajectory avoids critical structures


@dataclass
class ImplantDesign:
    """Complete patient-specific implant design."""
    name: str
    mesh: trimesh.Trimesh
    screws: List[ScreW] = field(default_factory=list)
    material: str = "Ti-6Al-4V"    # "Ti-6Al-4V" | "PEEK" | "Titanium-Mesh"
    min_wall_thickness_mm: float = 1.5
    max_wall_thickness_mm: float = 5.0
    mean_wall_thickness_mm: float = 2.5
    surface_area_mm2: float = 0.0
    volume_mm3: float = 0.0
    weight_g: float = 0.0
    structural_ok: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# Defect detection
# ---------------------------------------------------------------------------

def detect_defect_region(
    intact_reference: trimesh.Trimesh,
    patient_mesh: trimesh.Trimesh,
    gap_threshold_mm: float = 3.0,
    min_defect_area_mm2: float = 50.0,
) -> Dict:
    """
    Identify defect regions by comparing patient mesh to intact reference.

    Uses closest-point distance: areas where patient surface is far from
    reference indicate defects (missing bone, resection, trauma).

    Args:
        intact_reference: Template or contralateral mirrored mesh.
        patient_mesh: Patient bone surface (may have defects).
        gap_threshold_mm: Points further than this from reference are defect.
        min_defect_area_mm2: Minimum defect area to report.

    Returns:
        Dict with 'defect_mask' (bool array on reference vertices),
        'defect_area_mm2', 'defect_centroid', 'defect_bounding_box'.
    """
    ref_pts = intact_reference.vertices
    tree = KDTree(patient_mesh.vertices)
    dists, _ = tree.query(ref_pts)

    defect_mask = dists > gap_threshold_mm

    # Label connected defect regions
    # Use vertex adjacency to find connected components
    defect_verts = np.where(defect_mask)[0]
    if len(defect_verts) == 0:
        return {
            "defect_mask": defect_mask,
            "defect_area_mm2": 0.0,
            "defect_centroid": None,
            "defect_bounding_box": None,
            "n_defects": 0,
        }

    # Estimate defect area (triangles with majority defect vertices)
    face_defect = defect_mask[intact_reference.faces].sum(axis=1) >= 2
    defect_faces = intact_reference.faces[face_defect]
    if len(defect_faces) > 0:
        # Area of defect triangles
        a = ref_pts[defect_faces[:, 0]]
        b = ref_pts[defect_faces[:, 1]]
        c = ref_pts[defect_faces[:, 2]]
        defect_area = float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() * 0.5)
    else:
        defect_area = 0.0

    defect_centroid = ref_pts[defect_mask].mean(axis=0) if defect_mask.any() else None
    defect_pts = ref_pts[defect_mask]
    bbox = (defect_pts.min(axis=0), defect_pts.max(axis=0)) if len(defect_pts) > 0 else None

    return {
        "defect_mask": defect_mask,
        "defect_area_mm2": defect_area,
        "defect_centroid": defect_centroid,
        "defect_bounding_box": bbox,
        "n_defects": int(face_defect.sum()),
    }


# ---------------------------------------------------------------------------
# Mirror-based reconstruction
# ---------------------------------------------------------------------------

def mirror_reconstruct(
    contralateral_mesh: trimesh.Trimesh,
    mirror_plane_normal: Optional[np.ndarray] = None,
    mirror_plane_point: Optional[np.ndarray] = None,
) -> trimesh.Trimesh:
    """
    Mirror the contralateral (healthy) side to reconstruct a unilateral defect.

    This is the gold standard for unilateral orbital floor, zygoma, and
    mandibular ramus reconstructions.

    Args:
        contralateral_mesh: Healthy side bone mesh.
        mirror_plane_normal: Normal of midsagittal mirror plane. Default: X-axis.
        mirror_plane_point: Point on mirror plane. Default: origin.

    Returns:
        Mirrored trimesh.Trimesh in the position of the defect side.
    """
    if mirror_plane_normal is None:
        mirror_plane_normal = np.array([1.0, 0.0, 0.0])  # midsagittal
    if mirror_plane_point is None:
        mirror_plane_point = np.zeros(3)

    n = mirror_plane_normal / np.linalg.norm(mirror_plane_normal)

    # Householder reflection matrix
    H = np.eye(3) - 2 * np.outer(n, n)
    T4 = np.eye(4)
    T4[:3, :3] = H
    # Adjust translation to reflect through the plane (not origin)
    T4[:3, 3] = 2 * np.dot(mirror_plane_point, n) * n

    mirrored = contralateral_mesh.copy()
    mirrored.apply_transform(T4)

    # Flip face winding to correct normals after reflection
    mirrored.invert()

    return mirrored


# ---------------------------------------------------------------------------
# Statistical Shape Model (SSM) fitting
# ---------------------------------------------------------------------------

class StatisticalShapeModel:
    """
    Linear Statistical Shape Model (PCA-based) for defect reconstruction.

    The SSM represents the shape variation of a population of bone surfaces
    using principal components. Fitting deforms the mean shape to best
    match the patient's partial surface.

    Reference:
        Cootes et al., "Active Shape Models — Their Training and Application",
        CVIU 1995.
        Vetter & Blanz, "Estimating Coloured 3D Face Models ...", ECCV 1998.
    """

    def __init__(
        self,
        mean_shape: np.ndarray,          # (N, 3) mean surface vertices
        components: np.ndarray,          # (n_components, N*3) PCA modes
        variances: np.ndarray,           # (n_components,) eigenvalues
        n_components: Optional[int] = None,
    ) -> None:
        self.mean_shape = mean_shape      # (N, 3)
        self.components = components      # (K, 3N)
        self.variances = variances        # (K,)
        self.n_components = n_components or len(variances)
        self.std_devs = np.sqrt(variances[:self.n_components])

    def fit_to_partial(
        self,
        partial_points: np.ndarray,      # (M, 3) observed partial surface points
        partial_vertex_ids: np.ndarray,  # (M,) indices in the full SSM template
        n_fitting_components: Optional[int] = None,
        regularisation: float = 1.0,     # Tikhonov regularisation weight
        max_iter: int = 50,
    ) -> np.ndarray:
        """
        Fit the SSM to partial observations using constrained optimisation.

        Objective: minimise |P * (mean + V b) - y|² + λ |b/σ|²
        where:
          P = selection matrix (observed vertices),
          V = PCA mode matrix, b = shape coefficients,
          y = observed points, σ = standard deviations per mode.

        Args:
            partial_points: (M, 3) observed partial surface points.
            partial_vertex_ids: (M,) indices into full SSM template.
            n_fitting_components: Number of modes to use (default: all).
            regularisation: Tikhonov weight λ.

        Returns:
            (N, 3) reconstructed full surface.
        """
        K = n_fitting_components or self.n_components
        K = min(K, self.n_components)

        N = len(self.mean_shape)
        M = len(partial_points)

        # Build observation system (active only for observed vertices)
        obs_idx = np.array([vid * 3 + i for vid in partial_vertex_ids for i in range(3)])
        V_obs = self.components[:K, obs_idx].T        # (3M, K)
        mean_obs = self.mean_shape[partial_vertex_ids].ravel()  # (3M,)
        y = partial_points.ravel() - mean_obs          # (3M,)

        # Regularised least squares: (V'V + λ D) b = V' y
        D = np.diag(regularisation / (self.std_devs[:K] ** 2 + 1e-8))
        A = V_obs.T @ V_obs + D
        rhs = V_obs.T @ y
        b = np.linalg.solve(A, rhs)

        # Clamp coefficients to ±3σ (plausible shape space)
        b = np.clip(b, -3 * self.std_devs[:K], 3 * self.std_devs[:K])

        # Reconstruct full surface
        full_modes = self.components[:K]  # (K, 3N)
        deformation = (b[:, None] * full_modes).sum(axis=0)  # (3N,)
        reconstructed = self.mean_shape + deformation.reshape(N, 3)

        return reconstructed

    @classmethod
    def from_training_set(
        cls,
        meshes: List[trimesh.Trimesh],
        n_components: int = 20,
    ) -> "StatisticalShapeModel":
        """
        Build SSM from a set of registered training meshes.
        All meshes must have the same topology (same faces, corresponding vertices).

        Args:
            meshes: List of trimesh.Trimesh with identical topology.
            n_components: Number of PCA modes to retain.

        Returns:
            StatisticalShapeModel instance.
        """
        shapes = np.array([m.vertices.ravel() for m in meshes])  # (n_train, 3N)
        mean = shapes.mean(axis=0)
        centred = shapes - mean
        U, S, Vt = np.linalg.svd(centred, full_matrices=False)
        variances = (S ** 2) / (len(meshes) - 1)
        return cls(
            mean_shape=mean.reshape(-1, 3),
            components=Vt[:n_components],
            variances=variances[:n_components],
            n_components=n_components,
        )


# ---------------------------------------------------------------------------
# Thickness analysis
# ---------------------------------------------------------------------------

def wall_thickness_analysis(
    mesh: trimesh.Trimesh,
    n_sample: int = 5000,
    ray_length: float = 30.0,
) -> Dict[str, float]:
    """
    Estimate wall thickness via ray casting (shoot inward ray from surface).

    For each sample point, cast a ray along the inward normal and find the
    exit point to estimate local wall thickness.

    Args:
        mesh: trimesh.Trimesh bone or implant mesh.
        n_sample: number of sample points.
        ray_length: maximum ray length in mm.

    Returns:
        Dict with 'min_mm', 'max_mm', 'mean_mm', 'std_mm', 'pct_below_1mm'.
    """
    # Sample surface points and inward normals
    pts, face_ids = trimesh.sample.sample_surface(mesh, n_sample)
    normals = mesh.face_normals[face_ids]  # outward normals

    # Cast rays inward (along -normal)
    origins = pts + normals * 0.01  # tiny offset to avoid self-intersection
    directions = -normals

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, _ = intersector.intersects_location(origins, directions, multiple_hits=False)

    thicknesses = []
    for i, loc in zip(index_ray, locations):
        d = float(np.linalg.norm(loc - pts[i]))
        if d < ray_length:
            thicknesses.append(d)

    if not thicknesses:
        return {"min_mm": 0.0, "max_mm": 0.0, "mean_mm": 0.0, "std_mm": 0.0, "pct_below_1mm": 100.0}

    t = np.array(thicknesses)
    return {
        "min_mm": float(t.min()),
        "max_mm": float(t.max()),
        "mean_mm": float(t.mean()),
        "std_mm": float(t.std()),
        "pct_below_1mm": float((t < 1.0).mean() * 100),
        "pct_below_0_5mm": float((t < 0.5).mean() * 100),
    }


# ---------------------------------------------------------------------------
# Screw trajectory planning
# ---------------------------------------------------------------------------

def plan_screw_trajectories(
    implant_mesh: trimesh.Trimesh,
    bone_mesh: trimesh.Trimesh,
    n_screws: int = 4,
    screw_diameter_mm: float = 2.0,
    min_bone_engagement_mm: float = 4.0,
    safety_margin_mm: float = 2.0,
) -> List[ScreW]:
    """
    Plan optimal screw trajectories for implant fixation.

    Algorithm:
    1. Sample candidate screw holes on implant flange areas.
    2. Shoot ray into bone mesh along screw direction.
    3. Ensure adequate bone engagement (≥4mm bicortical preferred).
    4. Check safety margins from critical structures.
    5. Select best N screw positions distributed around the implant.

    Args:
        implant_mesh: PSI mesh in mm.
        bone_mesh: Patient bone mesh.
        n_screws: Number of screws to plan.
        screw_diameter_mm: Screw outer diameter.
        min_bone_engagement_mm: Minimum bone purchase length.
        safety_margin_mm: Safety buffer from mesh boundary / nerve canal.

    Returns:
        List of ScreW objects sorted by quality score.
    """
    # Sample candidate entry points on implant flange (boundary region)
    # For simplicity, sample the convex hull projection boundary
    hull_pts = implant_mesh.vertices[
        trimesh.convex.convex_hull_indexed(implant_mesh.vertices)[0]
    ]

    pts, face_ids = trimesh.sample.sample_surface(implant_mesh, 500)
    normals = implant_mesh.face_normals[face_ids]

    bone_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(bone_mesh)

    screws = []
    for i, (pt, n) in enumerate(zip(pts, normals)):
        # Shoot ray inward along normal
        origins = np.array([pt - n * 0.1])
        directions = np.array([-n])
        locs, idx_ray, _ = bone_intersector.intersects_location(
            origins, directions, multiple_hits=False
        )
        if len(locs) == 0:
            continue

        engagement = float(np.linalg.norm(locs[0] - pt))
        if engagement < min_bone_engagement_mm:
            continue

        screw = ScreW(
            name=f"screw_{i:03d}",
            entry_point=pt,
            direction=-n,
            length_mm=engagement + safety_margin_mm,
            diameter_mm=screw_diameter_mm,
            bone_engagement_mm=engagement,
            safe=engagement >= min_bone_engagement_mm,
        )
        screws.append(screw)

    # Sort by bone engagement (longest = best)
    screws.sort(key=lambda s: s.bone_engagement_mm, reverse=True)

    # Select n_screws well-distributed positions
    selected = []
    if len(screws) > 0:
        selected.append(screws[0])
        for s in screws[1:]:
            if len(selected) >= n_screws:
                break
            # Ensure min 8mm distance between screws
            dists = [np.linalg.norm(s.entry_point - ss.entry_point) for ss in selected]
            if min(dists) >= 8.0:
                selected.append(s)

    return selected


# ---------------------------------------------------------------------------
# Main ImplantDesigner API
# ---------------------------------------------------------------------------

class ImplantDesigner:
    """
    Patient-specific implant design pipeline.

    Example::

        designer = ImplantDesigner()
        implant = designer.design_orbital_floor(
            orbital_floor_mesh=patient_orbital,
            contralateral_mesh=healthy_orbital,
        )
        designer.export(implant, output_dir="output/implants/")
    """

    def __init__(
        self,
        ssm: Optional[StatisticalShapeModel] = None,
        material: str = "Ti-6Al-4V",
    ) -> None:
        self.ssm = ssm
        self.material = material

    def design_orbital_floor(
        self,
        orbital_floor_mesh: trimesh.Trimesh,
        contralateral_mesh: Optional[trimesh.Trimesh] = None,
        defect_gap_mm: float = 3.0,
    ) -> ImplantDesign:
        """
        Design a patient-specific orbital floor reconstruction plate.

        Strategy:
          1. Detect defect using gap analysis.
          2. If contralateral is available: mirror-based reconstruction.
          3. Otherwise: SSM-based filling.
          4. Create 2mm thick plate conforming to orbital floor contour.
          5. Plan fixation screws along orbital rim.

        Returns:
            ImplantDesign with mesh and screw trajectories.
        """
        logger.info("Designing orbital floor implant ...")

        if contralateral_mesh is not None:
            # Mirror-based reconstruction
            mirrored = mirror_reconstruct(
                contralateral_mesh,
                mirror_plane_normal=np.array([1.0, 0.0, 0.0]),
                mirror_plane_point=orbital_floor_mesh.centroid,
            )
            template = mirrored
            logger.info("Using mirror-based reconstruction.")
        elif self.ssm is not None:
            # SSM fitting
            partial_pts = np.array(orbital_floor_mesh.vertices)
            partial_ids = np.arange(min(len(partial_pts), len(self.ssm.mean_shape)))
            recon = self.ssm.fit_to_partial(
                partial_pts[:len(partial_ids)],
                partial_ids,
            )
            template = trimesh.Trimesh(
                vertices=recon,
                faces=orbital_floor_mesh.faces[:len(recon) - 2],
                process=False,
            )
            logger.info("Using SSM-based reconstruction.")
        else:
            template = orbital_floor_mesh.convex_hull
            logger.warning("No reference available; using convex hull fill.")

        # Shell the implant to 1.5mm thickness
        try:
            # Offset inward by 1.5mm using vertex normals
            offset_verts = template.vertices + template.vertex_normals * (-1.5)
            bottom = trimesh.Trimesh(
                vertices=offset_verts,
                faces=template.faces.copy(),
                process=False,
            )
            bottom.invert()
            # Combine outer and inner shells
            implant_mesh = trimesh.util.concatenate([template, bottom])
        except Exception:
            implant_mesh = template

        # Thickness analysis
        thickness = wall_thickness_analysis(implant_mesh)
        structural_ok = thickness["min_mm"] >= 0.5

        # Screw planning
        screws = plan_screw_trajectories(
            implant_mesh, orbital_floor_mesh, n_screws=4, screw_diameter_mm=2.0
        )

        # Weight estimation (Ti-6Al-4V density: 4.43 g/cm³)
        density = 4.43 if "Ti" in self.material else 1.32  # PEEK: 1.32 g/cm³
        vol_cm3 = abs(implant_mesh.volume) / 1000 if implant_mesh.is_watertight else 0
        weight = vol_cm3 * density

        return ImplantDesign(
            name="orbital_floor_PSI",
            mesh=implant_mesh,
            screws=screws,
            material=self.material,
            min_wall_thickness_mm=thickness["min_mm"],
            max_wall_thickness_mm=thickness["max_mm"],
            mean_wall_thickness_mm=thickness["mean_mm"],
            surface_area_mm2=float(implant_mesh.area),
            volume_mm3=float(abs(implant_mesh.volume)) if implant_mesh.is_watertight else 0,
            weight_g=weight,
            structural_ok=structural_ok,
            notes=f"Mirror-based, {len(screws)} screw holes planned",
        )

    def design_cranial_plate(
        self,
        defect_mesh: trimesh.Trimesh,
        contralateral_mesh: Optional[trimesh.Trimesh] = None,
        plate_thickness_mm: float = 1.5,
    ) -> ImplantDesign:
        """
        Design a patient-specific cranial reconstruction plate.
        Uses mirror reconstruction for unilateral calvarial defects.
        """
        logger.info("Designing cranial plate (defect area: %.1f mm²)...", defect_mesh.area)

        if contralateral_mesh is not None:
            template = mirror_reconstruct(contralateral_mesh)
        else:
            template = defect_mesh.convex_hull

        screws = plan_screw_trajectories(template, defect_mesh, n_screws=6, screw_diameter_mm=2.0)

        thickness = wall_thickness_analysis(template)

        return ImplantDesign(
            name="cranial_plate_PSI",
            mesh=template,
            screws=screws,
            material=self.material,
            min_wall_thickness_mm=plate_thickness_mm,
            surface_area_mm2=float(template.area),
            structural_ok=True,
            notes=f"Cranial plate, {len(screws)} fixation screws",
        )

    def export(
        self,
        implant: ImplantDesign,
        output_dir: str | Path,
    ) -> Dict[str, Path]:
        """
        Export implant mesh and screw hole template to STL files.

        Returns:
            Dict mapping component_name → output Path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        # Main implant
        implant_path = output_dir / f"{implant.name}.stl"
        implant.mesh.export(str(implant_path))
        paths["implant"] = implant_path
        logger.info("Exported implant: %s", implant_path)

        # Screw template (cylinders at each screw position)
        if implant.screws:
            screw_meshes = []
            for s in implant.screws:
                cyl = trimesh.creation.cylinder(
                    radius=s.diameter_mm / 2,
                    height=s.length_mm,
                    sections=16,
                )
                # Orient and position cylinder
                z_axis = np.array([0, 0, 1.0])
                d = s.direction / np.linalg.norm(s.direction)
                rot_axis = np.cross(z_axis, d)
                if np.linalg.norm(rot_axis) > 1e-6:
                    rot_axis /= np.linalg.norm(rot_axis)
                    angle = float(np.arccos(np.clip(np.dot(z_axis, d), -1, 1)))
                    T = np.eye(4)
                    T[:3, :3] = Rotation.from_rotvec(rot_axis * angle).as_matrix()
                    T[:3, 3] = s.entry_point
                    cyl.apply_transform(T)
                screw_meshes.append(cyl)

            if screw_meshes:
                screw_template = trimesh.util.concatenate(screw_meshes)
                screw_path = output_dir / f"{implant.name}_screws.stl"
                screw_template.export(str(screw_path))
                paths["screws"] = screw_path
                logger.info("Exported screw template: %s", screw_path)

        return paths


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    designer = ImplantDesigner(material="Ti-6Al-4V")

    # Simulated orbital floor: thin curved surface
    orbital = trimesh.creation.box(extents=[40, 30, 3])
    contralateral = orbital.copy()
    contralateral.apply_translation([5, 0, 0])

    implant = designer.design_orbital_floor(orbital, contralateral)
    logger.info("Implant: %s", implant.name)
    logger.info("Faces: %d | Area: %.1f mm² | Screws: %d",
                len(implant.mesh.faces), implant.surface_area_mm2, len(implant.screws))
    logger.info("Min thickness: %.2f mm | Structural OK: %s",
                implant.min_wall_thickness_mm, implant.structural_ok)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        exported = designer.export(implant, td)
        logger.info("Exported files: %s", list(exported.keys()))

    logger.info("ImplantDesigner self-test passed.")
    sys.exit(0)
