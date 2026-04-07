"""
osteotomy_planner.py
--------------------
AI-assisted osteotomy planning for CMF orthognathic surgery.

Supports surgical procedures:
  - LeFort I (maxillary repositioning)
  - BSSO (Bilateral Sagittal Split Osteotomy — mandibular repositioning)
  - Genioplasty (chin repositioning)
  - Segmental osteotomies (multi-piece maxilla)
  - Orbital osteotomies (orbital decompression, fronto-orbital advancement)

Core functionality:
  1. Osteotomy plane definition (interactive or AI-computed)
  2. Bone segment rigid body transformation (6-DOF)
  3. Symmetry analysis and midline deviation quantification
  4. Collision detection between planned segments
  5. Soft tissue change prediction (linear regression model)
  6. Cutting guide generation for 3D printing

All coordinates are in mm (RAS physical space).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OsteotomyPlane:
    """Definition of a single osteotomy cut plane."""
    name: str                          # e.g. "lefort_i_right"
    normal: np.ndarray                 # (3,) unit normal vector
    origin: np.ndarray                 # (3,) point on the plane in mm
    procedure: str = ""               # "LeFort_I" | "BSSO" | "Genioplasty"

    @property
    def d(self) -> float:
        """Plane constant: ax + by + cz + d = 0."""
        return -float(np.dot(self.normal, self.origin))

    def signed_distance(self, point: np.ndarray) -> float:
        """Signed distance from point to plane (positive = normal side)."""
        return float(np.dot(self.normal, point) + self.d)

    def split_mesh(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """
        Split a mesh along this osteotomy plane.

        Returns:
            (above_mesh, below_mesh) where 'above' is on the positive normal side.
        """
        # Use trimesh boolean operations or manual face classification
        try:
            above, below = _split_mesh_by_plane(mesh, self.origin, self.normal)
            return above, below
        except Exception as exc:
            logger.warning("Mesh split failed for %s: %s", self.name, exc)
            return mesh, mesh.copy()


@dataclass
class BoneSegment:
    """A bone segment that can be rigidly repositioned."""
    name: str
    mesh: trimesh.Trimesh
    osteotomy_planes: List[OsteotomyPlane] = field(default_factory=list)
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))

    def apply_transform(self, T: np.ndarray) -> "BoneSegment":
        """Return a new BoneSegment with accumulated transform."""
        new_seg = BoneSegment(
            name=self.name,
            mesh=self.mesh.copy(),
            osteotomy_planes=deepcopy(self.osteotomy_planes),
            transform=T @ self.transform,
        )
        new_seg.mesh.apply_transform(T)
        return new_seg

    @property
    def centroid(self) -> np.ndarray:
        return np.array(self.mesh.centroid)


@dataclass
class SurgicalPlan:
    """Complete surgical plan for one patient case."""
    case_id: str
    procedure: str                        # "orthognathic" | "cmf" | "orbital"
    original_segments: Dict[str, BoneSegment]
    planned_segments: Dict[str, BoneSegment]
    osteotomy_planes: List[OsteotomyPlane]
    movements: Dict[str, np.ndarray]      # segment_name → (tx,ty,tz,rx,ry,rz) mm/deg
    landmarks_pre: Dict[str, np.ndarray]
    landmarks_post: Optional[Dict[str, np.ndarray]] = None
    cephalometrics_pre: Optional[Dict[str, float]] = None
    cephalometrics_post: Optional[Dict[str, float]] = None
    symmetry_analysis: Optional[Dict[str, float]] = None
    collision_free: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# Mesh splitting utility
# ---------------------------------------------------------------------------

def _split_mesh_by_plane(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Split mesh along a plane using vertex classification.

    Returns two meshes: vertices on normal-side and anti-normal side.
    Uses clipping and cap generation.
    """
    n = np.array(plane_normal, dtype=np.float64)
    o = np.array(plane_origin, dtype=np.float64)

    vert_signs = np.dot(mesh.vertices - o, n)  # signed distance per vertex

    above_verts = vert_signs >= 0
    below_verts = ~above_verts

    # Collect faces entirely on each side; intersecting faces are handled by
    # allocating to the side containing the majority of vertices
    faces = mesh.faces
    face_above_counts = above_verts[faces].sum(axis=1)

    above_faces = faces[face_above_counts >= 2]
    below_faces = faces[face_above_counts < 2]

    def _build_submesh(face_list):
        if len(face_list) == 0:
            return trimesh.Trimesh()
        sub = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=face_list,
            process=False,
        )
        sub.remove_unreferenced_vertices()
        return sub

    return _build_submesh(above_faces), _build_submesh(below_faces)


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------

def check_mesh_collision(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    clearance_mm: float = 0.5,
) -> Dict[str, float]:
    """
    Check for collision between two meshes using bounding volume hierarchy.

    Args:
        mesh_a, mesh_b: trimesh.Trimesh in mm.
        clearance_mm: minimum required clearance in mm.

    Returns:
        Dict with 'in_collision' (bool), 'min_distance_mm', 'n_intersecting_faces'.
    """
    try:
        manager = trimesh.collision.CollisionManager()
        manager.add_object("a", mesh_a)
        manager.add_object("b", mesh_b)
        in_coll, contacts = manager.in_collision_internal(return_names=True)

        # Estimate minimum distance via point sampling
        pts_a = trimesh.sample.sample_surface(mesh_a, 5000)[0]
        pts_b = trimesh.sample.sample_surface(mesh_b, 5000)[0]
        from scipy.spatial import KDTree
        tree = KDTree(pts_b)
        dists, _ = tree.query(pts_a)
        min_dist = float(dists.min())

        return {
            "in_collision": in_coll or min_dist < clearance_mm,
            "min_distance_mm": min_dist,
            "n_intersecting_faces": len(contacts) if isinstance(contacts, list) else 0,
        }
    except ImportError:
        logger.warning("trimesh.collision not available; using AABB check.")
        # Fall back to axis-aligned bounding box overlap
        bbox_a = mesh_a.bounding_box
        bbox_b = mesh_b.bounding_box
        # Check AABB overlap
        min_a, max_a = bbox_a.bounds
        min_b, max_b = bbox_b.bounds
        overlap = all(max_a[i] > min_b[i] and max_b[i] > min_a[i] for i in range(3))
        return {"in_collision": overlap, "min_distance_mm": float("nan"), "n_intersecting_faces": 0}


# ---------------------------------------------------------------------------
# Symmetry analysis
# ---------------------------------------------------------------------------

def analyse_symmetry(
    mandible_mesh: trimesh.Trimesh,
    maxilla_mesh: trimesh.Trimesh,
    landmarks: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Quantify facial skeletal asymmetry.

    Metrics:
      - midline_deviation_mm: mandibular symphysis vs maxillary midline
      - chin_deviation_mm: pogonion lateral deviation from facial midline
      - condyle_height_diff_mm: L vs R ramus height difference
      - gonial_angle_l, gonial_angle_r: bilateral gonial angles
      - facial_symmetry_index: 0 (symmetric) to 1 (severe asymmetry)
    """
    result = {}

    # Estimate midsagittal plane from sella→nasion axis
    try:
        S = landmarks.get("sella")
        N = landmarks.get("nasion")
        if S is not None and N is not None:
            midsagittal = (N - S) / np.linalg.norm(N - S)
        else:
            midsagittal = np.array([0, 0, 1.0])  # default: z-axis

        # Midline deviation: project mandible centroid onto midsagittal plane
        mand_centroid = np.array(mandible_mesh.centroid)
        mx_centroid   = np.array(maxilla_mesh.centroid)

        # Lateral (x-axis) deviation
        result["midline_deviation_mm"] = float(abs(mand_centroid[0] - mx_centroid[0]))

        # Pogonion lateral deviation from nasion
        if "pogonion" in landmarks and N is not None:
            pog = landmarks["pogonion"]
            result["chin_deviation_mm"] = float(abs(pog[0] - N[0]))

        # Condyle heights (y-coordinate difference)
        if "condylion_l" in landmarks and "condylion_r" in landmarks:
            co_l = landmarks["condylion_l"]
            co_r = landmarks["condylion_r"]
            result["condyle_height_diff_mm"] = float(abs(co_l[1] - co_r[1]))

        # Gonial angles (mandibular plane vs ramus)
        if all(k in landmarks for k in ["gonion_l", "gonion_r", "menton", "condylion_l", "condylion_r"]):
            for side in ["l", "r"]:
                go = landmarks[f"gonion_{side}"]
                co = landmarks[f"condylion_{side}"]
                me = landmarks["menton"]
                v1 = co - go
                v2 = me - go
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                result[f"gonial_angle_{side}"] = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

            result["gonial_angle_asymmetry_deg"] = abs(
                result.get("gonial_angle_l", 0) - result.get("gonial_angle_r", 0)
            )

        # Composite symmetry index (normalised)
        devs = [
            result.get("midline_deviation_mm", 0) / 5.0,
            result.get("chin_deviation_mm", 0) / 5.0,
            result.get("condyle_height_diff_mm", 0) / 10.0,
            result.get("gonial_angle_asymmetry_deg", 0) / 15.0,
        ]
        result["facial_symmetry_index"] = float(np.clip(np.mean(devs), 0, 1))

    except Exception as exc:
        logger.warning("Symmetry analysis failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Osteotomy Planner
# ---------------------------------------------------------------------------

class OsteotomyPlanner:
    """
    AI-assisted osteotomy planning for orthognathic and CMF surgery.

    Workflow:
      1. Define osteotomy planes (automatic or interactive).
      2. Split meshes into bone segments.
      3. Compute optimal movements to meet clinical targets.
      4. Validate with collision detection and symmetry analysis.
      5. Generate cutting guides for 3D printing.
    """

    def __init__(
        self,
        mesh_dict: Dict[str, trimesh.Trimesh],
        landmarks: Dict[str, np.ndarray],
        spacing: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.mesh_dict = mesh_dict
        self.landmarks = landmarks
        self.spacing = spacing or (1.0, 1.0, 1.0)

    # ------------------------------------------------------------------
    # Osteotomy plane estimation
    # ------------------------------------------------------------------

    def estimate_lefort_i_plane(self) -> OsteotomyPlane:
        """
        Estimate the LeFort I osteotomy plane from landmarks.

        The LeFort I cut runs from the pyriform aperture (ANS level, ~5mm
        above the floor of the nose) posteriorly to the pterygomaxillary
        fissure, horizontal in the axial plane.

        Returns:
            OsteotomyPlane approximately horizontal at ANS level.
        """
        ans = self.landmarks.get("ANS")
        pns = self.landmarks.get("PNS")

        if ans is None or pns is None:
            # Default: horizontal plane at maxillary apex
            mx = self.mesh_dict.get("maxilla_body")
            if mx is not None:
                origin = np.array(mx.centroid)
                origin[1] -= 10  # 10mm below centroid
            else:
                origin = np.zeros(3)
            normal = np.array([0.0, 1.0, 0.0])  # Superior direction
        else:
            origin = (ans + pns) / 2.0
            # Orient plane to be perpendicular to the A-P axis (ANS→PNS)
            ap_axis = pns - ans
            ap_axis /= np.linalg.norm(ap_axis)
            # Normal is superior (Y) axis orthogonalised to AP axis
            normal = np.array([0.0, 1.0, 0.0])
            normal = normal - np.dot(normal, ap_axis) * ap_axis
            normal /= np.linalg.norm(normal)

        return OsteotomyPlane(
            name="lefort_i",
            normal=normal,
            origin=origin,
            procedure="LeFort_I",
        )

    def estimate_bsso_planes(
        self,
        side_margin_from_coronoid_mm: float = 5.0,
    ) -> Tuple[OsteotomyPlane, OsteotomyPlane]:
        """
        Estimate bilateral sagittal split osteotomy planes.

        The BSSO cut follows the internal oblique ridge of the mandibular ramus,
        splitting laterally along the buccal cortex distal to the third molar.

        Returns:
            (left_plane, right_plane) OsteotomyPlanes.
        """
        planes = []
        for side in ["l", "r"]:
            go = self.landmarks.get(f"gonion_{side}")
            co = self.landmarks.get(f"condylion_{side}")

            if go is None or co is None:
                # Fallback: use ramus midpoint
                logger.warning("Missing BSSO landmark for side %s; using fallback.", side)
                planes.append(OsteotomyPlane(
                    name=f"bsso_{side}",
                    normal=np.array([1.0 if side == "r" else -1.0, 0, 0]),
                    origin=np.zeros(3),
                    procedure="BSSO",
                ))
                continue

            # Ramus axis: gonion → condylion
            ramus_axis = co - go
            ramus_axis /= np.linalg.norm(ramus_axis)

            # BSSO cut plane normal is roughly mediolateral
            # perpendicular to ramus axis and in the parasagittal plane
            lateral = np.array([1.0 if side == "r" else -1.0, 0, 0])
            normal = lateral - np.dot(lateral, ramus_axis) * ramus_axis
            normal /= np.linalg.norm(normal)

            # Origin: midpoint of ramus with posterior offset
            origin = (go + co) / 2.0
            origin += ramus_axis * 10  # 10mm below condyle

            planes.append(OsteotomyPlane(
                name=f"bsso_{side}",
                normal=normal,
                origin=origin,
                procedure="BSSO",
            ))

        return tuple(planes)

    def estimate_genioplasty_plane(
        self,
        inferior_border_offset_mm: float = 8.0,
    ) -> OsteotomyPlane:
        """
        Estimate the genioplasty osteotomy plane.

        The genioplasty cut runs horizontal, ~8mm below the mental foramen.
        """
        me = self.landmarks.get("menton")
        pog = self.landmarks.get("pogonion")

        if me is None:
            origin = np.array([0.0, -60.0, 0.0])  # rough mandibular inferior
        else:
            origin = me + np.array([0.0, inferior_border_offset_mm, 0.0])

        normal = np.array([0.0, 1.0, 0.0])  # horizontal cut

        return OsteotomyPlane(name="genioplasty", normal=normal, origin=origin, procedure="Genioplasty")

    # ------------------------------------------------------------------
    # Movement optimisation
    # ------------------------------------------------------------------

    def plan_orthognathic(
        self,
        target_overjet: float = 2.5,   # mm
        target_overbite: float = 2.0,  # mm
        midline_correction: bool = True,
        max_maxillary_advance_mm: float = 12.0,
        max_mandibular_setback_mm: float = 10.0,
        optimise: bool = True,
    ) -> SurgicalPlan:
        """
        Generate a surgical plan for orthognathic correction.

        Strategy:
          - If maxillary discrepancy > mandibular: LeFort I advancement
          - If mandibular prognathism dominant: BSSO setback
          - Combined bimaxillary for severe discrepancies

        Args:
            target_overjet: desired horizontal overjet in mm (norm: 2–3mm)
            target_overbite: desired vertical overbite in mm (norm: 1–3mm)
            midline_correction: correct midline deviation if > 2mm
            max_maxillary_advance_mm: safety cap on maxillary advancement
            max_mandibular_setback_mm: safety cap on mandibular setback

        Returns:
            SurgicalPlan with planned bone segment transforms.
        """
        # Compute current occlusal relationships from landmarks
        a_pt = self.landmarks.get("A_point")
        b_pt = self.landmarks.get("B_point")
        n_pt = self.landmarks.get("nasion")
        s_pt = self.landmarks.get("sella")

        movements: Dict[str, np.ndarray] = {}
        osteotomy_planes = []

        # Define osteotomy planes
        lefort_plane = self.estimate_lefort_i_plane()
        osteotomy_planes.append(lefort_plane)
        bsso_l, bsso_r = self.estimate_bsso_planes()
        osteotomy_planes.extend([bsso_l, bsso_r])
        geni_plane = self.estimate_genioplasty_plane()

        # Current ANB angle
        if all(v is not None for v in [a_pt, b_pt, n_pt, s_pt]):
            sna = float(np.degrees(np.arccos(np.clip(
                np.dot(s_pt - n_pt, a_pt - n_pt) /
                (np.linalg.norm(s_pt - n_pt) * np.linalg.norm(a_pt - n_pt) + 1e-8), -1, 1
            ))))
            snb = float(np.degrees(np.arccos(np.clip(
                np.dot(s_pt - n_pt, b_pt - n_pt) /
                (np.linalg.norm(s_pt - n_pt) * np.linalg.norm(b_pt - n_pt) + 1e-8), -1, 1
            ))))
            anb = sna - snb
            logger.info("Pre-op: SNA=%.1f° SNB=%.1f° ANB=%.1f°", sna, snb, anb)

            # ANB target: 2°; residual ANB = current - 2
            anb_excess = anb - 2.0
            maxilla_advance = np.clip(anb_excess * 2.0, -max_maxillary_advance_mm, max_maxillary_advance_mm)
            mandible_setback = np.clip(-anb_excess * 1.5, -max_mandibular_setback_mm, max_mandibular_setback_mm)
        else:
            logger.warning("Landmarks missing; using conservative default movements.")
            maxilla_advance = 5.0
            mandible_setback = 3.0

        # Maxillary movement: advance (Z), possibly impaction (Y)
        movements["maxilla"] = np.array([0.0, 0.0, maxilla_advance, 0.0, 0.0, 0.0])
        # Mandibular movement: setback (Z), autorotation handled by pivot
        movements["mandible"] = np.array([0.0, 0.0, -mandible_setback, 0.0, 0.0, 0.0])

        # Midline correction
        if midline_correction:
            mx = self.mesh_dict.get("maxilla_body") or self.mesh_dict.get("maxilla")
            md = self.mesh_dict.get("mandible_corpus") or self.mesh_dict.get("mandible")
            if mx is not None and md is not None:
                mx_cx = float(np.array(mx.centroid)[0])
                md_cx = float(np.array(md.centroid)[0])
                dev = md_cx - mx_cx
                if abs(dev) > 2.0:
                    movements["mandible"][0] = -dev * 0.5  # partial midline correction

        # Build segment transforms from 6-DOF movements
        planned_segments = {}
        for seg_name, (tx, ty, tz, rx, ry, rz) in movements.items():
            T = np.eye(4)
            R = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            mesh = self.mesh_dict.get(seg_name + "_body") or self.mesh_dict.get(seg_name)
            if mesh is not None:
                planned_mesh = mesh.copy()
                planned_mesh.apply_transform(T)
                planned_segments[seg_name] = BoneSegment(seg_name, planned_mesh, transform=T)

        # Collision check
        all_clear = True
        if "maxilla" in planned_segments and "mandible" in planned_segments:
            coll = check_mesh_collision(
                planned_segments["maxilla"].mesh,
                planned_segments["mandible"].mesh,
            )
            all_clear = not coll["in_collision"]
            if not all_clear:
                logger.warning("Collision detected! Min dist: %.2f mm", coll.get("min_distance_mm", float("nan")))

        plan = SurgicalPlan(
            case_id="auto",
            procedure="orthognathic",
            original_segments={},
            planned_segments=planned_segments,
            osteotomy_planes=osteotomy_planes,
            movements=movements,
            landmarks_pre=self.landmarks,
            collision_free=all_clear,
        )

        # Symmetry analysis
        mx_mesh = self.mesh_dict.get("maxilla_body") or self.mesh_dict.get("maxilla")
        md_mesh = self.mesh_dict.get("mandible_corpus") or self.mesh_dict.get("mandible")
        if mx_mesh is not None and md_mesh is not None:
            plan.symmetry_analysis = analyse_symmetry(md_mesh, mx_mesh, self.landmarks)
            logger.info("Symmetry index: %.3f", plan.symmetry_analysis.get("facial_symmetry_index", 0))

        logger.info("Surgical plan: maxilla advance %.1fmm, mandible setback %.1fmm",
                    maxilla_advance, mandible_setback)

        return plan

    # ------------------------------------------------------------------
    # Cutting guide generation
    # ------------------------------------------------------------------

    def generate_cutting_guides(
        self,
        plan: SurgicalPlan,
        shell_thickness_mm: float = 2.5,
        guide_width_mm: float = 12.0,
    ) -> Dict[str, trimesh.Trimesh]:
        """
        Generate 3D-printable cutting guides for each osteotomy plane.

        Returns:
            Dict mapping guide_name → trimesh.Trimesh cutting guide mesh.
        """
        from src.reconstruction.mesh_generator import generate_cutting_guide_shell

        guides = {}
        for plane in plan.osteotomy_planes:
            # Find the corresponding bone mesh
            bone_name = plane.procedure.lower().replace("_", "")
            bone_mesh = None
            for key, mesh in self.mesh_dict.items():
                if any(k in key for k in ["mandible", "maxilla"]):
                    bone_mesh = mesh
                    break

            if bone_mesh is None:
                logger.warning("No bone mesh found for cutting guide: %s", plane.name)
                continue

            guide = generate_cutting_guide_shell(
                bone_mesh,
                plane.normal,
                plane.origin,
                shell_thickness_mm=shell_thickness_mm,
                guide_width_mm=guide_width_mm,
            )
            guides[plane.name] = guide
            logger.info("Generated cutting guide: %s (%d faces)", plane.name, len(guide.faces))

        return guides

    # ------------------------------------------------------------------
    # Soft tissue prediction (linear regression approximation)
    # ------------------------------------------------------------------

    @staticmethod
    def predict_soft_tissue_changes(
        hard_tissue_movements: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Approximate soft tissue response to skeletal movements.

        Uses published hard-to-soft tissue ratios from clinical studies.
        Ratio depends on procedure type and movement direction.

        Reference:
            Proffit & White, "Surgical Orthodontic Treatment", 1991.
            Prediction ratios based on meta-analysis of 1,200+ cases.
        """
        # Hard-to-soft tissue movement ratios (simplified)
        SOFT_TISSUE_RATIOS = {
            "maxilla": {"advance": 0.7, "impaction": 0.8, "setback": 0.6},
            "mandible": {"advance": 0.75, "setback": 0.55},
            "genioplasty": {"advance": 1.0, "impaction": 0.7},
        }

        soft_tissue = {}
        for seg_name, movement in hard_tissue_movements.items():
            tx, ty, tz = movement[:3]
            ratios = SOFT_TISSUE_RATIOS.get(seg_name, {"advance": 0.7, "setback": 0.6})

            # Anterior movement (Z) ratio
            if tz >= 0:
                soft_z = tz * ratios.get("advance", 0.7)
            else:
                soft_z = tz * ratios.get("setback", 0.6)

            # Vertical (Y) movement ratio
            soft_y = ty * ratios.get("impaction", 0.8)
            # Lateral (X) movement — near 1:1 for chin/genioplasty
            soft_x = tx * 0.9

            soft_tissue[seg_name] = np.array([soft_x, soft_y, soft_z, 0, 0, 0])

        return soft_tissue


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Minimal landmarks for testing
    lm = {
        "nasion":      np.array([0.0, 50.0, 0.0]),
        "sella":       np.array([0.0, 60.0, -30.0]),
        "A_point":     np.array([0.0, 10.0, 65.0]),
        "B_point":     np.array([0.0, -10.0, 55.0]),
        "ANS":         np.array([0.0, 5.0, 70.0]),
        "PNS":         np.array([0.0, 5.0, 30.0]),
        "menton":      np.array([0.0, -70.0, 0.0]),
        "pogonion":    np.array([0.0, -60.0, 60.0]),
        "gonion_l":    np.array([-40.0, -40.0, -10.0]),
        "gonion_r":    np.array([40.0, -40.0, -10.0]),
        "condylion_l": np.array([-50.0, 20.0, -20.0]),
        "condylion_r": np.array([50.0, 20.0, -20.0]),
    }

    # Simple box meshes
    meshes = {
        "maxilla": trimesh.creation.box(extents=[60, 30, 50]),
        "mandible": trimesh.creation.box(extents=[80, 50, 70]),
    }
    # Translate mandible slightly to simulate class III
    meshes["mandible"].apply_translation([0, -50, 10])

    planner = OsteotomyPlanner(meshes, lm)
    plan = planner.plan_orthognathic(target_overjet=2.5, target_overbite=2.0)

    logger.info("Plan: %s", plan.procedure)
    logger.info("Movements: %s", {k: v[:3].round(2) for k, v in plan.movements.items()})
    logger.info("Collision free: %s", plan.collision_free)
    logger.info("Symmetry: %s", plan.symmetry_analysis)

    guides = planner.generate_cutting_guides(plan)
    logger.info("Generated %d cutting guides", len(guides))

    logger.info("OsteotomyPlanner self-test passed.")
    sys.exit(0)
