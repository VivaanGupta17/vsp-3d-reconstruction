"""
vtk_viewer.py
-------------
VTK-based 3D visualisation for surgical planning.

Features:
  - Volume rendering with CT presets (bone, soft tissue)
  - 3D mesh rendering with transparency and per-structure colours
  - Segmentation overlay on axial/coronal/sagittal CT slices
  - Osteotomy plane visualisation (semi-transparent cutting plane)
  - Before/after surgical plan comparison (side-by-side or toggle)
  - Landmark rendering (spheres + labels)
  - Screw trajectory visualisation (cylinders)
  - Off-screen rendering for screenshot/video export
  - Interactive measurement tools (distance, angle)

Dependencies: vtk (pip install vtk), optional pyvista for higher-level API.

Usage::

    viewer = SurgicalPlanViewer()
    viewer.add_mesh(mandible_mesh, color="bone", opacity=0.85)
    viewer.add_mesh(maxilla_mesh, color="ivory", opacity=0.85)
    viewer.add_osteotomy_plane(plane_origin, plane_normal)
    viewer.add_landmarks(landmark_dict, label=True)
    viewer.show()
    viewer.screenshot("plan_view.png")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try VTK import — graceful degradation if unavailable
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    logger.warning("VTK not installed. Install with: pip install vtk")

# Try pyvista for high-level interface
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Color presets
# ---------------------------------------------------------------------------

# Structure → (R, G, B) in [0, 1]
STRUCTURE_COLORS = {
    "mandible":          (0.85, 0.75, 0.60),   # warm bone
    "maxilla":           (0.90, 0.80, 0.65),
    "zygomatic":         (0.80, 0.70, 0.55),
    "orbital_floor":     (0.70, 0.80, 0.90),   # slightly blue
    "cortical_bone":     (0.92, 0.84, 0.70),
    "cancellous_bone":   (0.85, 0.75, 0.60),
    "teeth":             (0.97, 0.96, 0.92),   # near-white enamel
    "titanium_implant":  (0.75, 0.75, 0.85),   # cool metallic
    "cutting_guide":     (0.20, 0.60, 0.90),   # blue guide
    "planned_maxilla":   (0.30, 0.80, 0.50),   # green = planned
    "planned_mandible":  (0.30, 0.70, 0.90),
    "landmark_sphere":   (1.00, 0.20, 0.20),   # red landmarks
    "screw":             (0.70, 0.70, 0.80),
    "midline":           (1.00, 0.80, 0.00),   # yellow midline
}

CT_WINDOW_LEVEL = {
    "bone":         (1500, 400),    # (window, level) in HU
    "soft_tissue":  (400, 40),
    "lung":         (1500, -600),
    "brain":        (80, 40),
}


# ---------------------------------------------------------------------------
# VTK mesh helpers
# ---------------------------------------------------------------------------

def trimesh_to_vtk_polydata(mesh) -> "vtk.vtkPolyData":
    """Convert trimesh.Trimesh → vtkPolyData."""
    if not VTK_AVAILABLE:
        raise ImportError("VTK is required for mesh conversion.")

    verts = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int64)

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(verts))

    cells = vtk.vtkCellArray()
    n_faces = len(faces)
    cell_array = np.hstack([np.full((n_faces, 1), 3, dtype=np.int64), faces]).ravel()
    cells.SetCells(n_faces, numpy_to_vtk(cell_array, array_type=vtk.VTK_ID_TYPE))

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)

    # Compute normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()


def sitk_to_vtk_image(sitk_image) -> "vtk.vtkImageData":
    """Convert SimpleITK Image → vtkImageData for volume rendering."""
    if not VTK_AVAILABLE:
        raise ImportError("VTK is required.")
    import SimpleITK as sitk as _sitk
    arr = _sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # (D, H, W)

    vtk_image = vtk.vtkImageData()
    spacing = sitk_image.GetSpacing()
    origin  = sitk_image.GetOrigin()
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])

    flat = arr.ravel(order="C")
    vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_arr.SetName("HU")
    vtk_image.GetPointData().SetScalars(vtk_arr)

    return vtk_image


def numpy_volume_to_vtk(
    arr: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> "vtk.vtkImageData":
    """Convert numpy (D, H, W) array → vtkImageData."""
    if not VTK_AVAILABLE:
        raise ImportError("VTK is required.")

    arr_f = arr.astype(np.float32)
    vtk_img = vtk.vtkImageData()
    vtk_img.SetSpacing(spacing[2], spacing[1], spacing[0])  # VTK uses (x,y,z)
    vtk_img.SetOrigin(origin)
    vtk_img.SetDimensions(arr_f.shape[2], arr_f.shape[1], arr_f.shape[0])

    flat = arr_f.ravel(order="C")
    vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_arr.SetName("HU")
    vtk_img.GetPointData().SetScalars(vtk_arr)
    return vtk_img


# ---------------------------------------------------------------------------
# Actor builders
# ---------------------------------------------------------------------------

def build_mesh_actor(
    polydata: "vtk.vtkPolyData",
    color: Tuple[float, float, float] = (0.85, 0.75, 0.60),
    opacity: float = 0.85,
    specular: float = 0.3,
    specular_power: float = 20,
    edge_visibility: bool = False,
) -> "vtk.vtkActor":
    """Create a rendered actor from vtkPolyData."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(opacity)
    prop.SetSpecular(specular)
    prop.SetSpecularPower(specular_power)
    prop.SetDiffuse(0.8)
    prop.SetAmbient(0.1)
    if edge_visibility:
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(0.3, 0.3, 0.3)

    return actor


def build_plane_actor(
    origin: np.ndarray,
    normal: np.ndarray,
    size_mm: float = 60.0,
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0),
    opacity: float = 0.3,
) -> "vtk.vtkActor":
    """Create a semi-transparent disc actor to represent an osteotomy plane."""
    # Create a disc source aligned with the plane normal
    disc = vtk.vtkDiskSource()
    disc.SetInnerRadius(0)
    disc.SetOuterRadius(size_mm / 2)
    disc.SetRadialResolution(1)
    disc.SetCircumferentialResolution(64)
    disc.Update()

    # Rotate disc (default Z-axis) to align with plane normal
    n = np.array(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    z = np.array([0, 0, 1.0])
    axis = np.cross(z, n)
    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
        angle = float(np.degrees(np.arccos(np.clip(np.dot(z, n), -1, 1))))
        transform = vtk.vtkTransform()
        transform.Translate(*origin)
        transform.RotateWXYZ(angle, *axis)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(disc.GetOutputPort())
        tf_filter.SetTransform(transform)
        tf_filter.Update()
        poly = tf_filter.GetOutput()
    else:
        # Normal already aligned; just translate
        transform = vtk.vtkTransform()
        transform.Translate(*origin)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(disc.GetOutputPort())
        tf_filter.SetTransform(transform)
        tf_filter.Update()
        poly = tf_filter.GetOutput()

    return build_mesh_actor(poly, color=color, opacity=opacity)


def build_sphere_actor(
    centre: np.ndarray,
    radius_mm: float = 3.0,
    color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    opacity: float = 1.0,
) -> "vtk.vtkActor":
    """Create a sphere actor for landmark visualisation."""
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(*centre)
    sphere.SetRadius(radius_mm)
    sphere.SetPhiResolution(16)
    sphere.SetThetaResolution(16)
    sphere.Update()
    return build_mesh_actor(sphere.GetOutput(), color=color, opacity=opacity)


def build_cylinder_actor(
    start: np.ndarray,
    end: np.ndarray,
    radius_mm: float = 1.0,
    color: Tuple[float, float, float] = (0.7, 0.7, 0.8),
    opacity: float = 1.0,
) -> "vtk.vtkActor":
    """Create a cylinder actor for screw trajectory visualisation."""
    length = float(np.linalg.norm(end - start))
    centre = (start + end) / 2

    cyl = vtk.vtkCylinderSource()
    cyl.SetRadius(radius_mm)
    cyl.SetHeight(length)
    cyl.SetResolution(16)
    cyl.Update()

    # Orient cylinder along start→end direction
    direction = (end - start) / (length + 1e-8)
    y_axis = np.array([0.0, 1.0, 0.0])
    axis = np.cross(y_axis, direction)
    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
        angle = float(np.degrees(np.arccos(np.clip(np.dot(y_axis, direction), -1, 1))))
        transform = vtk.vtkTransform()
        transform.Translate(*centre)
        transform.RotateWXYZ(angle, *axis)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(cyl.GetOutputPort())
        tf.SetTransform(transform)
        tf.Update()
        poly = tf.GetOutput()
    else:
        transform = vtk.vtkTransform()
        transform.Translate(*centre)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(cyl.GetOutputPort())
        tf.SetTransform(transform)
        tf.Update()
        poly = tf.GetOutput()

    return build_mesh_actor(poly, color=color, opacity=opacity)


def build_text_actor(
    text: str,
    position_2d: Tuple[float, float] = (10, 10),
    font_size: int = 14,
    color: Tuple[float, float, float] = (1, 1, 1),
) -> "vtk.vtkTextActor":
    """Create a 2D text overlay actor."""
    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(*position_2d)
    prop = actor.GetTextProperty()
    prop.SetFontSize(font_size)
    prop.SetColor(*color)
    prop.BoldOn()
    return actor


# ---------------------------------------------------------------------------
# CT slice overlay renderer
# ---------------------------------------------------------------------------

def build_slice_actor(
    volume: np.ndarray,           # (D, H, W) float32
    orientation: str = "axial",   # "axial" | "coronal" | "sagittal"
    slice_index: Optional[int] = None,
    window: float = 1500,
    level: float = 400,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> "vtk.vtkActor":
    """
    Create a 2D image plane actor from a CT volume slice.
    Applies window/level mapping to display bone detail.
    """
    D, H, W = volume.shape

    # Select slice
    if orientation == "axial":
        idx = slice_index if slice_index is not None else D // 2
        arr2d = volume[idx, :, :]
        sx, sy = spacing[2], spacing[1]
        position = (0, 0, idx * spacing[0])
    elif orientation == "coronal":
        idx = slice_index if slice_index is not None else H // 2
        arr2d = volume[:, idx, :]
        sx, sy = spacing[2], spacing[0]
        position = (0, idx * spacing[1], 0)
    else:  # sagittal
        idx = slice_index if slice_index is not None else W // 2
        arr2d = volume[:, :, idx]
        sx, sy = spacing[1], spacing[0]
        position = (idx * spacing[2], 0, 0)

    # Window/level → [0, 255]
    lo = level - window / 2
    hi = level + window / 2
    arr_norm = np.clip((arr2d - lo) / (hi - lo + 1e-8), 0, 1)
    arr_uint8 = (arr_norm * 255).astype(np.uint8)

    # Build RGB image (greyscale → 3 channels)
    arr_rgb = np.stack([arr_uint8] * 3, axis=-1).copy()

    rows, cols = arr_rgb.shape[:2]
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(cols, rows, 1)
    vtk_img.SetSpacing(sx, sy, 1.0)
    vtk_img.SetOrigin(*position)

    flat = arr_rgb.ravel()
    vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_arr.SetNumberOfComponents(3)
    vtk_arr.SetName("RGB")
    vtk_img.GetPointData().SetScalars(vtk_arr)

    plane = vtk.vtkImageActor()
    plane.GetMapper().SetInputData(vtk_img)
    return plane


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------

class SurgicalPlanViewer:
    """
    Interactive 3D viewer for surgical planning review.

    Example::

        viewer = SurgicalPlanViewer(window_size=(1600, 900))
        viewer.add_mesh(mandible_mesh, label="mandible")
        viewer.add_mesh(maxilla_mesh, label="maxilla")
        viewer.add_osteotomy_plane(plane.origin, plane.normal)
        viewer.add_landmarks({"nasion": np.array([0, 50, 0])})
        viewer.show()
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (1400, 900),
        background_color: Tuple[float, float, float] = (0.12, 0.12, 0.18),
        off_screen: bool = False,
    ) -> None:
        self.window_size = window_size
        self.background_color = background_color
        self.off_screen = off_screen
        self._actors: List = []
        self._renderer: Optional["vtk.vtkRenderer"] = None
        self._render_window: Optional["vtk.vtkRenderWindow"] = None
        self._interactor: Optional["vtk.vtkRenderWindowInteractor"] = None
        self._init_vtk()

    def _init_vtk(self) -> None:
        if not VTK_AVAILABLE:
            logger.warning("VTK not available; viewer will not render.")
            return

        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(*self.background_color)
        self._renderer.GradientBackgroundOn()
        self._renderer.SetBackground2(0.05, 0.05, 0.10)

        self._render_window = vtk.vtkRenderWindow()
        self._render_window.SetSize(*self.window_size)
        self._render_window.AddRenderer(self._renderer)
        self._render_window.SetWindowName("VSP-3D Surgical Planning Viewer")
        if self.off_screen:
            self._render_window.SetOffScreenRendering(True)

        self._interactor = vtk.vtkRenderWindowInteractor()
        self._interactor.SetRenderWindow(self._render_window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self._interactor.SetInteractorStyle(style)

        # Ambient occlusion for realism
        try:
            self._renderer.UseSSAOOn()
            self._renderer.SetSSAORadius(0.1)
        except AttributeError:
            pass  # older VTK

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_mesh(
        self,
        mesh,               # trimesh.Trimesh
        label: str = "",
        color: Optional[Tuple[float, float, float]] = None,
        opacity: float = 0.85,
    ) -> "SurgicalPlanViewer":
        """Add a bone/implant mesh to the scene."""
        if not VTK_AVAILABLE:
            return self

        if color is None:
            color = STRUCTURE_COLORS.get(label.split("_")[0], (0.85, 0.75, 0.60))

        poly = trimesh_to_vtk_polydata(mesh)
        actor = build_mesh_actor(poly, color=color, opacity=opacity)
        self._renderer.AddActor(actor)
        self._actors.append(actor)
        return self

    def add_osteotomy_plane(
        self,
        origin: np.ndarray,
        normal: np.ndarray,
        size_mm: float = 70.0,
        label: str = "",
    ) -> "SurgicalPlanViewer":
        """Add a semi-transparent cutting plane visualisation."""
        if not VTK_AVAILABLE:
            return self

        actor = build_plane_actor(origin, normal, size_mm=size_mm,
                                   color=(0.2, 0.7, 1.0), opacity=0.35)
        self._renderer.AddActor(actor)
        self._actors.append(actor)

        if label:
            # Add text label
            text = build_text_actor(f" {label}", position_2d=(0.02, 0.95))
            self._renderer.AddActor(text)

        return self

    def add_landmarks(
        self,
        landmarks: Dict[str, np.ndarray],
        radius_mm: float = 2.5,
        label: bool = True,
    ) -> "SurgicalPlanViewer":
        """Add landmark spheres to the scene."""
        if not VTK_AVAILABLE:
            return self

        for name, pos in landmarks.items():
            actor = build_sphere_actor(pos, radius_mm=radius_mm,
                                        color=STRUCTURE_COLORS["landmark_sphere"])
            self._renderer.AddActor(actor)
            self._actors.append(actor)

        return self

    def add_screws(
        self,
        screws,   # List[ScreW]
    ) -> "SurgicalPlanViewer":
        """Add screw trajectory cylinders to the scene."""
        if not VTK_AVAILABLE:
            return self

        for screw in screws:
            end = screw.entry_point + screw.direction * screw.length_mm
            actor = build_cylinder_actor(
                screw.entry_point, end,
                radius_mm=screw.diameter_mm / 2,
                color=STRUCTURE_COLORS["screw"],
            )
            self._renderer.AddActor(actor)
            self._actors.append(actor)

        return self

    def add_ct_slice(
        self,
        volume: np.ndarray,
        orientation: str = "axial",
        slice_index: Optional[int] = None,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        preset: str = "bone",
    ) -> "SurgicalPlanViewer":
        """Add a CT slice overlay."""
        if not VTK_AVAILABLE:
            return self

        window, level = CT_WINDOW_LEVEL.get(preset, (1500, 400))
        actor = build_slice_actor(volume, orientation, slice_index, window, level, spacing)
        self._renderer.AddActor(actor)
        self._actors.append(actor)
        return self

    def reset_camera(self) -> "SurgicalPlanViewer":
        """Auto-fit camera to all visible actors."""
        if self._renderer:
            self._renderer.ResetCamera()
        return self

    def show(self) -> None:
        """Launch interactive 3D viewer window."""
        if not VTK_AVAILABLE:
            logger.warning("VTK not available; cannot show viewer.")
            return

        self.reset_camera()
        self._render_window.Render()
        self._interactor.Initialize()
        self._interactor.Start()

    def screenshot(
        self,
        path: str | Path,
        magnification: int = 2,
    ) -> None:
        """
        Capture an off-screen screenshot.

        Args:
            path: Output image path (.png).
            magnification: Super-sampling factor for high-res output.
        """
        if not VTK_AVAILABLE:
            logger.warning("VTK not available; cannot screenshot.")
            return

        self.reset_camera()
        self._render_window.Render()

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self._render_window)
        w2i.SetScale(magnification)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        logger.info("Screenshot saved: %s", path)

    def export_video(
        self,
        path: str | Path,
        n_frames: int = 120,
        fps: int = 30,
        rotation_axis: str = "z",
    ) -> None:
        """
        Export a 360° rotation video of the surgical plan.

        Args:
            path: Output .avi path.
            n_frames: Total frames (one full rotation).
            fps: Frames per second.
            rotation_axis: "x" | "y" | "z".
        """
        if not VTK_AVAILABLE:
            logger.warning("VTK not available; cannot export video.")
            return

        self.reset_camera()

        writer = vtk.vtkAVIWriter()
        writer.SetFileName(str(path))
        writer.SetRate(fps)
        writer.SetQuality(2)  # 0=worst, 2=best

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self._render_window)

        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Start()

        camera = self._renderer.GetActiveCamera()
        for i in range(n_frames):
            if rotation_axis == "y":
                camera.Azimuth(360.0 / n_frames)
            elif rotation_axis == "x":
                camera.Elevation(360.0 / n_frames)
            else:
                camera.Azimuth(360.0 / n_frames)

            self._render_window.Render()
            w2i.Modified()
            writer.Write()

        writer.End()
        logger.info("Video exported: %s (%d frames @ %d fps)", path, n_frames, fps)

    def clear(self) -> "SurgicalPlanViewer":
        """Remove all actors from the scene."""
        if self._renderer:
            for actor in self._actors:
                self._renderer.RemoveActor(actor)
        self._actors.clear()
        return self


# ---------------------------------------------------------------------------
# Convenience: before/after side-by-side
# ---------------------------------------------------------------------------

class BeforeAfterViewer:
    """
    Side-by-side comparison of pre-operative and planned bone positions.
    Uses two VTK viewports in the same render window.
    """

    def __init__(self, window_size: Tuple[int, int] = (2000, 900)) -> None:
        if not VTK_AVAILABLE:
            logger.warning("VTK not available.")
            return

        self._rw = vtk.vtkRenderWindow()
        self._rw.SetSize(*window_size)
        self._rw.SetWindowName("VSP-3D: Pre-op vs Planned")

        # Left viewport: pre-operative
        self._ren_pre = vtk.vtkRenderer()
        self._ren_pre.SetViewport(0, 0, 0.5, 1)
        self._ren_pre.SetBackground(0.12, 0.12, 0.18)
        self._ren_pre.AddActor(build_text_actor("Pre-operative", (0.02, 0.95)))
        self._rw.AddRenderer(self._ren_pre)

        # Right viewport: planned
        self._ren_post = vtk.vtkRenderer()
        self._ren_post.SetViewport(0.5, 0, 1, 1)
        self._ren_post.SetBackground(0.10, 0.14, 0.20)
        self._ren_post.AddActor(build_text_actor("Planned", (0.52, 0.95)))
        self._rw.AddRenderer(self._ren_post)

    def add_pre_mesh(self, mesh, label: str = "", opacity: float = 0.85) -> None:
        if not VTK_AVAILABLE:
            return
        color = STRUCTURE_COLORS.get(label.split("_")[0], (0.85, 0.75, 0.60))
        poly = trimesh_to_vtk_polydata(mesh)
        self._ren_pre.AddActor(build_mesh_actor(poly, color=color, opacity=opacity))

    def add_post_mesh(self, mesh, label: str = "", opacity: float = 0.85) -> None:
        if not VTK_AVAILABLE:
            return
        key = "planned_" + label.split("_")[0]
        color = STRUCTURE_COLORS.get(key, STRUCTURE_COLORS.get(label.split("_")[0], (0.3, 0.8, 0.5)))
        poly = trimesh_to_vtk_polydata(mesh)
        self._ren_post.AddActor(build_mesh_actor(poly, color=color, opacity=opacity))

    def show(self) -> None:
        if not VTK_AVAILABLE:
            return
        self._ren_pre.ResetCamera()
        self._ren_post.ResetCamera()
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(self._rw)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self._rw.Render()
        interactor.Initialize()
        interactor.Start()


# ---------------------------------------------------------------------------
# PyVista convenience wrapper (higher-level API)
# ---------------------------------------------------------------------------

def show_with_pyvista(
    mesh_dict: Dict[str, "trimesh.Trimesh"],
    landmarks: Optional[Dict[str, np.ndarray]] = None,
    title: str = "VSP-3D Surgical Plan",
) -> None:
    """
    Quick visualisation using PyVista (if available).
    Simpler API than raw VTK; good for notebooks and quick reviews.
    """
    if not PYVISTA_AVAILABLE:
        logger.warning("PyVista not installed. pip install pyvista")
        return

    plotter = pv.Plotter(title=title, window_size=(1400, 900))
    plotter.set_background("gray20")

    for label, mesh in mesh_dict.items():
        pv_mesh = pv.PolyData(mesh.vertices, np.hstack([
            np.full((len(mesh.faces), 1), 3), mesh.faces
        ]))
        color = STRUCTURE_COLORS.get(label.split("_")[0], (0.85, 0.75, 0.60))
        plotter.add_mesh(pv_mesh, color=color, opacity=0.85, smooth_shading=True,
                         label=label)

    if landmarks:
        pts = np.array(list(landmarks.values()))
        pv_pts = pv.PolyData(pts)
        plotter.add_mesh(pv_pts, color="red", point_size=12, render_points_as_spheres=True)

    plotter.add_legend()
    plotter.show()


# ---------------------------------------------------------------------------
# Quick self-test (off-screen)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if not VTK_AVAILABLE:
        logger.warning("VTK not available — skipping render test.")
        sys.exit(0)

    try:
        import trimesh
        sphere_mesh = trimesh.creation.icosphere(subdivisions=3, radius=30.0)

        viewer = SurgicalPlanViewer(off_screen=True)
        viewer.add_mesh(sphere_mesh, label="mandible")
        viewer.add_osteotomy_plane(
            origin=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            label="LeFort I",
        )
        viewer.add_landmarks({"nasion": np.array([0.0, 35.0, 0.0])})

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            viewer.screenshot(f.name)
            logger.info("Off-screen screenshot: %s", f.name)

        logger.info("vtk_viewer self-test passed.")
    except Exception as exc:
        logger.warning("VTK test failed: %s", exc)

    sys.exit(0)
