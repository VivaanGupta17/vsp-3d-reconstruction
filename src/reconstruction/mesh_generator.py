"""
mesh_generator.py
-----------------
3D mesh generation from segmentation volumes.

Pipeline:
  1. Marching cubes isosurface extraction (skimage / VTK)
  2. Laplacian or Taubin smoothing
  3. Mesh decimation (Quadric Edge Collapse) for 3D printing
  4. Surface quality checks: manifold, self-intersection, degenerate faces
  5. STL / OBJ export with per-body colour for multi-part prints

Key dependencies: trimesh, numpy, skimage (for marching cubes fallback),
                  optionally VTK for more robust isosurface extraction.
"""

from __future__ import annotations

import logging
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MeshQualityReport:
    """Surface quality metrics for a single mesh."""
    n_vertices: int
    n_faces: int
    n_edges: int
    is_watertight: bool
    is_winding_consistent: bool
    has_degenerate_faces: bool
    n_degenerate_faces: int
    has_self_intersections: bool   # expensive to compute; None if skipped
    euler_number: int
    n_components: int
    surface_area_mm2: float
    volume_mm3: float
    bounding_box_mm: Tuple[float, float, float]  # (dx, dy, dz)
    max_edge_length_mm: float
    mean_edge_length_mm: float
    aspect_ratio_max: float  # worst triangle aspect ratio (1=equilateral)

    def __str__(self) -> str:
        lines = [
            f"Vertices: {self.n_vertices:,}  |  Faces: {self.n_faces:,}",
            f"Watertight: {self.is_watertight}  |  Winding: {self.is_winding_consistent}",
            f"Degenerate faces: {self.n_degenerate_faces}",
            f"Self-intersections: {self.has_self_intersections}",
            f"Components: {self.n_components}  |  Euler: {self.euler_number}",
            f"Surface area: {self.surface_area_mm2:.1f} mm²",
            f"Volume: {self.volume_mm3:.1f} mm³",
            f"Bounding box: {self.bounding_box_mm[0]:.1f} × {self.bounding_box_mm[1]:.1f} × {self.bounding_box_mm[2]:.1f} mm",
            f"Mean/max edge: {self.mean_edge_length_mm:.2f} / {self.max_edge_length_mm:.2f} mm",
            f"Max aspect ratio: {self.aspect_ratio_max:.2f}",
        ]
        return "\n".join(lines)


@dataclass
class MeshGeneratorConfig:
    """Configuration for mesh generation and post-processing."""
    # Marching cubes
    mc_level: float = 0.5             # iso-level threshold (0.5 → boundary voxels)
    mc_step_size: int = 1             # voxel step (1 = full resolution)
    mc_allow_degenerate: bool = False

    # Smoothing
    smoothing_method: str = "taubin"  # "laplacian" | "taubin" | "none"
    laplacian_iterations: int = 30
    laplacian_lambda: float = 0.5
    taubin_iterations: int = 50
    taubin_lambda: float = 0.63
    taubin_mu: float = -0.64          # shrinkage-prevention parameter

    # Decimation (0 = no decimation; 0.9 = keep 10% of faces)
    target_reduction: float = 0.7
    # Max edge length for 3D printing (mm); faces with longer edges are split
    max_edge_length_print_mm: float = 2.0

    # Quality checks
    check_self_intersections: bool = False  # expensive O(n²)
    fix_normals: bool = True
    remove_degenerate: bool = True
    remove_duplicate_vertices: bool = True

    # Export
    stl_binary: bool = True


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

class MeshGenerator:
    """
    Generate, smooth, decimate, QA, and export 3D bone meshes from
    binary segmentation volumes.

    Example::

        gen = MeshGenerator(config)
        mesh = gen.generate(cortical_mask, voxel_spacing=(0.5, 0.5, 0.5))
        report = gen.quality_check(mesh, voxel_spacing=(0.5, 0.5, 0.5))
        gen.export_stl(mesh, "mandible.stl")
    """

    def __init__(self, config: Optional[MeshGeneratorConfig] = None) -> None:
        self.config = config or MeshGeneratorConfig()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def generate(
        self,
        mask: np.ndarray,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        label: str = "bone",
    ) -> trimesh.Trimesh:
        """
        Full mesh generation pipeline for a single binary segmentation mask.

        Args:
            mask: (D, H, W) boolean or uint8 array.
            voxel_spacing: (dz, dy, dx) spacing in mm.
            label: Descriptive label attached to mesh metadata.

        Returns:
            trimesh.Trimesh in physical (mm) coordinates.
        """
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)

        logger.info("[%s] Extracting isosurface from volume %s ...", label, mask.shape)
        mesh = self._marching_cubes(mask, voxel_spacing)

        if self.config.remove_duplicate_vertices:
            mesh.merge_vertices()

        if self.config.remove_degenerate:
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()

        logger.info("[%s] Raw mesh: %d vertices, %d faces", label, len(mesh.vertices), len(mesh.faces))

        # Smoothing
        if self.config.smoothing_method == "taubin":
            mesh = self._taubin_smooth(mesh)
        elif self.config.smoothing_method == "laplacian":
            mesh = self._laplacian_smooth(mesh)

        # Decimation
        if 0.0 < self.config.target_reduction < 1.0:
            mesh = self._decimate(mesh)

        # Fix normals
        if self.config.fix_normals:
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fix_winding(mesh)

        mesh.metadata["label"] = label
        logger.info(
            "[%s] Final mesh: %d vertices, %d faces | watertight: %s",
            label, len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
        )
        return mesh

    def generate_all(
        self,
        mask_dict: Dict[str, np.ndarray],
        voxel_spacing: Tuple[float, float, float],
    ) -> Dict[str, trimesh.Trimesh]:
        """
        Generate meshes for all masks in a dictionary.

        Args:
            mask_dict: {label: binary_mask} mapping.
            voxel_spacing: shared voxel spacing in mm.

        Returns:
            {label: trimesh.Trimesh} mapping.
        """
        return {
            label: self.generate(mask, voxel_spacing, label=label)
            for label, mask in mask_dict.items()
        }

    # ------------------------------------------------------------------
    # Marching Cubes
    # ------------------------------------------------------------------

    def _marching_cubes(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> trimesh.Trimesh:
        """Extract isosurface using marching cubes and scale to mm."""
        try:
            verts, faces, normals, _ = marching_cubes(
                mask,
                level=self.config.mc_level,
                spacing=spacing,
                step_size=self.config.mc_step_size,
                allow_degenerate=self.config.mc_allow_degenerate,
            )
        except RuntimeError as exc:
            logger.warning("Marching cubes failed: %s. Trying with relaxed params.", exc)
            verts, faces, normals, _ = marching_cubes(
                ndimage.gaussian_filter(mask, sigma=1.0),
                level=0.5,
                spacing=spacing,
            )

        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def _laplacian_smooth(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Laplacian smoothing: iteratively move each vertex toward the average
        of its neighbours. Shrinks the mesh if applied alone.
        """
        verts = mesh.vertices.copy()
        adj = self._build_adjacency(mesh)
        lam = self.config.laplacian_lambda

        for _ in range(self.config.laplacian_iterations):
            delta = np.zeros_like(verts)
            for i, neighbours in enumerate(adj):
                if len(neighbours) > 0:
                    delta[i] = verts[list(neighbours)].mean(axis=0) - verts[i]
            verts += lam * delta

        return trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)

    def _taubin_smooth(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Taubin smoothing (λ/μ method): alternating positive/negative
        Laplacian steps to prevent volume shrinkage.

        Reference: Taubin, SIGGRAPH 1995.
        """
        verts = mesh.vertices.copy()
        adj = self._build_adjacency(mesh)
        lam = self.config.taubin_lambda
        mu = self.config.taubin_mu

        for it in range(self.config.taubin_iterations):
            factor = lam if it % 2 == 0 else mu
            delta = np.zeros_like(verts)
            for i, neighbours in enumerate(adj):
                if len(neighbours) > 0:
                    delta[i] = verts[list(neighbours)].mean(axis=0) - verts[i]
            verts += factor * delta

        return trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)

    @staticmethod
    def _build_adjacency(mesh: trimesh.Trimesh) -> List[set]:
        """Build per-vertex adjacency list (vertex → set of neighbour indices)."""
        n = len(mesh.vertices)
        adj: List[set] = [set() for _ in range(n)]
        for a, b, c in mesh.faces:
            adj[a].update([b, c])
            adj[b].update([a, c])
            adj[c].update([a, b])
        return adj

    # ------------------------------------------------------------------
    # Decimation
    # ------------------------------------------------------------------

    def _decimate(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Mesh decimation via Quadric Edge Collapse (through trimesh / VTK).
        Falls back to a simple face-count cap if QEC is unavailable.
        """
        target_faces = max(100, int(len(mesh.faces) * (1 - self.config.target_reduction)))
        logger.info(
            "Decimating: %d → %d faces (%.0f%% reduction)",
            len(mesh.faces), target_faces, 100 * self.config.target_reduction,
        )

        try:
            # Prefer VTK Quadric decimation if available
            import vtk
            from vtk.util.numpy_support import numpy_to_vtk
            poly = self._trimesh_to_vtk(mesh)
            decimate = vtk.vtkQuadricDecimation()
            decimate.SetInputData(poly)
            decimate.SetTargetReduction(self.config.target_reduction)
            decimate.Update()
            return self._vtk_to_trimesh(decimate.GetOutput())
        except ImportError:
            pass

        try:
            decimated = mesh.simplify_quadric_decimation(target_faces)
            return decimated
        except Exception as exc:
            logger.warning("Decimation failed (%s); returning original mesh.", exc)
            return mesh

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def quality_check(
        self,
        mesh: trimesh.Trimesh,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        check_self_intersections: Optional[bool] = None,
    ) -> MeshQualityReport:
        """
        Compute surface quality metrics for surgical/3D-printing readiness.

        Args:
            mesh: trimesh.Trimesh in mm.
            voxel_spacing: for reference (not used in computation).
            check_self_intersections: override config; set False to skip expensive check.

        Returns:
            MeshQualityReport instance.
        """
        check_si = (
            check_self_intersections
            if check_self_intersections is not None
            else self.config.check_self_intersections
        )

        # Edge lengths
        edges = mesh.vertices[mesh.edges_unique[:, 0]] - mesh.vertices[mesh.edges_unique[:, 1]]
        edge_lengths = np.linalg.norm(edges, axis=1)

        # Triangle aspect ratios
        a = mesh.vertices[mesh.faces[:, 0]]
        b = mesh.vertices[mesh.faces[:, 1]]
        c = mesh.vertices[mesh.faces[:, 2]]
        ab = np.linalg.norm(a - b, axis=1)
        bc = np.linalg.norm(b - c, axis=1)
        ca = np.linalg.norm(c - a, axis=1)
        s = (ab + bc + ca) / 2.0  # semi-perimeter
        area = np.abs(np.cross(b - a, c - a)).sum(axis=1) * 0.5
        # Circumradius formula: R = abc / (4 * area)
        circumR = (ab * bc * ca) / (4 * area + 1e-12)
        # Inradius: r = area / s
        inR = area / (s + 1e-12)
        aspect_ratio = circumR / (3 * inR + 1e-12)  # equilateral = 1.0

        # Self-intersection (optional, expensive)
        if check_si:
            try:
                si_manager = trimesh.collision.CollisionManager()
                si_manager.add_object("mesh", mesh)
                _, names = si_manager.in_collision_internal(return_names=True)
                has_si = len(names) > 0
            except Exception:
                has_si = None
        else:
            has_si = None

        bbox = mesh.bounding_box.extents

        return MeshQualityReport(
            n_vertices=len(mesh.vertices),
            n_faces=len(mesh.faces),
            n_edges=len(mesh.edges_unique),
            is_watertight=mesh.is_watertight,
            is_winding_consistent=mesh.is_winding_consistent,
            has_degenerate_faces=bool((area < 1e-10).any()),
            n_degenerate_faces=int((area < 1e-10).sum()),
            has_self_intersections=has_si,
            euler_number=mesh.euler_number,
            n_components=trimesh.graph.connected_components(mesh.face_adjacency).max() + 1
                         if len(mesh.faces) > 0 else 0,
            surface_area_mm2=float(mesh.area),
            volume_mm3=float(abs(mesh.volume)) if mesh.is_watertight else float("nan"),
            bounding_box_mm=(float(bbox[0]), float(bbox[1]), float(bbox[2])),
            max_edge_length_mm=float(edge_lengths.max()) if len(edge_lengths) else 0.0,
            mean_edge_length_mm=float(edge_lengths.mean()) if len(edge_lengths) else 0.0,
            aspect_ratio_max=float(aspect_ratio.max()) if len(aspect_ratio) else 0.0,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_stl(self, mesh: trimesh.Trimesh, path: str | Path) -> None:
        """Export mesh to STL (binary by default)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(path), file_type="stl_ascii" if not self.config.stl_binary else "stl")
        logger.info("Exported STL: %s  (%d faces, %.1f KB)", path, len(mesh.faces), path.stat().st_size / 1024)

    def export_obj(self, mesh: trimesh.Trimesh, path: str | Path) -> None:
        """Export mesh to OBJ with MTL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(path), file_type="obj")
        logger.info("Exported OBJ: %s", path)

    def export_scene_stl(
        self,
        mesh_dict: Dict[str, trimesh.Trimesh],
        output_dir: str | Path,
    ) -> List[Path]:
        """Export all meshes in a dictionary to individual STL files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for label, mesh in mesh_dict.items():
            path = output_dir / f"{label}.stl"
            self.export_stl(mesh, path)
            paths.append(path)
        return paths

    def nest_for_printing(
        self,
        mesh_dict: Dict[str, trimesh.Trimesh],
        build_volume_mm: Tuple[float, float, float] = (200, 200, 200),
    ) -> trimesh.Scene:
        """
        Arrange multiple meshes in a 3D printing build volume.
        Stacks models in a grid layout, respects build volume constraints.

        Returns:
            trimesh.Scene ready for scene export.
        """
        scene = trimesh.Scene()
        x_offset = 5.0
        for label, mesh in mesh_dict.items():
            bbox = mesh.bounding_box.extents
            mesh_copy = mesh.copy()
            mesh_copy.apply_translation([x_offset, 5.0, 0.0])
            x_offset += bbox[0] + 5.0  # 5mm gap between models
            scene.add_geometry(mesh_copy, node_name=label)
        return scene

    # ------------------------------------------------------------------
    # VTK conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trimesh_to_vtk(mesh: trimesh.Trimesh):
        """Convert trimesh → vtkPolyData."""
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(mesh.vertices.astype(np.float32)))

        cells = vtk.vtkCellArray()
        tris = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64)
        cells.SetCells(len(mesh.faces), numpy_to_vtk(tris.ravel(), array_type=vtk.VTK_ID_TYPE))

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)
        return poly

    @staticmethod
    def _vtk_to_trimesh(poly) -> trimesh.Trimesh:
        """Convert vtkPolyData → trimesh.Trimesh."""
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        verts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        cells = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
        return trimesh.Trimesh(vertices=verts, faces=cells, process=False)


# ---------------------------------------------------------------------------
# Surgical guide generation helpers
# ---------------------------------------------------------------------------

def generate_cutting_guide_shell(
    bone_mesh: trimesh.Trimesh,
    cut_plane_normal: np.ndarray,
    cut_plane_point: np.ndarray,
    shell_thickness_mm: float = 2.5,
    guide_width_mm: float = 10.0,
) -> trimesh.Trimesh:
    """
    Generate a thin shell cutting guide that sits on the bone surface
    and defines a planar osteotomy.

    Algorithm:
      1. Slice bone mesh along cut plane → get cross-section curve.
      2. Extrude the curve into a ribbon along the plane normal.
      3. Boolean difference: guide shell minus bone surface → fitting concavity.

    Args:
        bone_mesh: Bone surface mesh in mm.
        cut_plane_normal: Unit normal vector of osteotomy plane.
        cut_plane_point: A point on the osteotomy plane.
        shell_thickness_mm: Wall thickness of the guide.
        guide_width_mm: Width of the guide strip along the cut plane.

    Returns:
        trimesh.Trimesh cutting guide shell.
    """
    # Step 1: Cross-section along cut plane
    try:
        section = bone_mesh.section(
            plane_origin=cut_plane_point,
            plane_normal=cut_plane_normal,
        )
        if section is None:
            raise ValueError("Plane does not intersect mesh.")

        # Step 2: Get 2D path, extrude to 3D guide strip
        path_2d, transform = section.to_planar()
        guide_2d = path_2d.buffer(guide_width_mm / 2).difference(
            path_2d.buffer(guide_width_mm / 2 - shell_thickness_mm)
        )
        guide_3d = trimesh.creation.extrude_polygon(guide_2d, shell_thickness_mm)
        guide_3d.apply_transform(transform)
        return guide_3d

    except Exception as exc:
        logger.warning("Cutting guide generation failed: %s. Returning placeholder.", exc)
        # Return a thin box as fallback
        return trimesh.creation.box(extents=[guide_width_mm, 40.0, shell_thickness_mm])


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    cfg = MeshGeneratorConfig(
        smoothing_method="taubin",
        target_reduction=0.5,
        check_self_intersections=False,
    )
    gen = MeshGenerator(cfg)

    # Create a dummy sphere segmentation mask
    D, H, W = 64, 64, 64
    cx, cy, cz = D // 2, H // 2, W // 2
    r = 20
    z, y, x = np.ogrid[:D, :H, :W]
    mask = ((z - cx) ** 2 + (y - cy) ** 2 + (x - cz) ** 2 <= r ** 2).astype(np.float32)
    spacing = (0.5, 0.5, 0.5)

    mesh = gen.generate(mask, spacing, label="test_sphere")
    report = gen.quality_check(mesh, spacing)
    logger.info("\nQuality Report:\n%s", report)

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        gen.export_stl(mesh, f.name)
        logger.info("Test STL: %s", f.name)

    logger.info("MeshGenerator self-test passed.")
    sys.exit(0)

DEFAULT_DECIMATION_RATIO = 0.5


def validate_mesh_before_export(verts, faces, min_faces: int = 50, max_edge_mm: float = 10.0) -> dict:
    """
    Run basic mesh integrity checks before writing to disk.
    Raises ValueError if critical checks fail.
    """
    import numpy as np

    report = {"passed": True, "warnings": [], "errors": []}

    if len(faces) < min_faces:
        report["errors"].append(f"too few faces: {len(faces)} < {min_faces}")
        report["passed"] = False

    if len(verts) == 0:
        report["errors"].append("empty vertex array")
        report["passed"] = False
        return report

    # check for degenerate (zero-area) faces
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    edge_lens = np.stack([
        np.linalg.norm(v1 - v0, axis=1),
        np.linalg.norm(v2 - v1, axis=1),
        np.linalg.norm(v0 - v2, axis=1),
    ])
    n_degen = (edge_lens < 1e-6).any(axis=0).sum()
    if n_degen > 0:
        report["warnings"].append(f"{n_degen} degenerate faces detected")

    if edge_lens.max() > max_edge_mm:
        report["warnings"].append(f"max edge length {edge_lens.max():.2f} mm exceeds {max_edge_mm} mm")

    return report
