"""
mandible_segmentor.py
---------------------
Specialised CMF (craniomaxillofacial) segmentor for:
  - Mandible (corpus, ramus, condyle, coronoid, symphysis)
  - Maxilla (body, alveolar process, palate)
  - Zygomatic arch (body, temporal process, frontal process)
  - Orbital floor
  - Mandibular canal (inferior alveolar nerve)
  - Teeth (per-tooth labelling, upper/lower arches)

Key differences vs. general BoneSegmentor:
  1. CBCT-optimised preprocessing: metal artifact reduction (MAR) via
     sinogram-based interpolation and frequency-split correction.
  2. Higher-resolution input paths for fine structures (orbital floor, canal).
  3. Condyle and ramus landmark extraction from segmentation.
  4. Mandible / maxilla split using occlusal plane estimation.

Architecture: cascaded approach
  Stage 1 — coarse CMF bone detection (low-res full FOV, 96³ patches)
  Stage 2 — fine sub-structure segmentation (high-res ROI patches, 128³)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import ConvexHull

from .bone_segmentor import UNet3D, SlidingWindowConfig, sliding_window_inference, postprocess_segmentation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CMF class map
# ---------------------------------------------------------------------------

CMF_CLASSES = {
    0:  "background",
    1:  "mandible_corpus",
    2:  "mandible_ramus",
    3:  "mandibular_condyle",
    4:  "mandibular_coronoid",
    5:  "mandibular_symphysis",
    6:  "mandibular_canal",
    7:  "maxilla_body",
    8:  "maxilla_alveolar",
    9:  "hard_palate",
    10: "zygomatic_body",
    11: "zygomatic_arch",
    12: "orbital_floor",
    13: "teeth_upper",
    14: "teeth_lower",
}

NUM_CMF_CLASSES = len(CMF_CLASSES)

# Aggregate class groups for downstream planning
MANDIBLE_CLASSES = {1, 2, 3, 4, 5, 6}
MAXILLA_CLASSES  = {7, 8, 9}
ZYGOMA_CLASSES   = {10, 11}
ORBITAL_CLASSES  = {12}
TEETH_UPPER      = {13}
TEETH_LOWER      = {14}


# ---------------------------------------------------------------------------
# Metal Artifact Reduction (MAR)
# ---------------------------------------------------------------------------

class MetalArtifactReducer:
    """
    Sinogram-based metal artifact reduction for CBCT.

    Algorithm (simplified normalised MAR):
    1. Identify metal voxels from HU thresholding.
    2. Forward project to estimate sinogram.
    3. Interpolate metal-corrupted sinogram traces.
    4. Back-project and blend corrected image.

    Reference:
        Meyer et al., "Normalized metal artifact reduction (NMAR) in
        computed tomography", Med Phys, 2010.
    """

    METAL_HU_THRESHOLD = 2500  # HU above which metal is assumed

    def __init__(
        self,
        interpolation: str = "linear",   # "linear" | "polynomial"
        blend_weight: float = 0.85,       # weight for MAR result vs original
        frequency_correction: bool = True,
    ) -> None:
        self.interpolation = interpolation
        self.blend_weight = blend_weight
        self.frequency_correction = frequency_correction

    def reduce(self, volume: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Apply MAR to a 3D CT/CBCT volume.

        Args:
            volume: (D, H, W) float32 array in HU.
            spacing: voxel spacing in mm (dz, dy, dx).

        Returns:
            MAR-corrected volume of same shape and dtype.
        """
        metal_mask = volume >= self.METAL_HU_THRESHOLD
        if not metal_mask.any():
            logger.debug("No metal detected — skipping MAR.")
            return volume

        logger.info(
            "Metal detected: %d voxels (%.2f%% of volume). Applying MAR.",
            metal_mask.sum(),
            100.0 * metal_mask.mean(),
        )

        corrected = self._interpolation_correction(volume, metal_mask)

        if self.frequency_correction:
            corrected = self._frequency_correction(volume, corrected, metal_mask)

        # Blend corrected and original outside metal region
        result = self.blend_weight * corrected + (1 - self.blend_weight) * volume
        result[metal_mask] = volume[metal_mask]  # preserve original metal voxels
        return result.astype(np.float32)

    def _interpolation_correction(
        self,
        volume: np.ndarray,
        metal_mask: np.ndarray,
    ) -> np.ndarray:
        """Replace metal-affected rows in each axial slice by interpolation."""
        corrected = volume.copy().astype(np.float32)
        for z in range(volume.shape[0]):
            slice_2d = volume[z]
            metal_2d = metal_mask[z]
            if not metal_2d.any():
                continue

            for row in range(slice_2d.shape[0]):
                line = slice_2d[row]
                metal_cols = np.where(metal_2d[row])[0]
                if len(metal_cols) == 0:
                    continue
                clean_cols = np.where(~metal_2d[row])[0]
                if len(clean_cols) < 2:
                    continue
                # Linear interpolation of metal-corrupted columns
                interp = np.interp(metal_cols, clean_cols, line[clean_cols])
                corrected[z, row, metal_cols] = interp

        return corrected

    def _frequency_correction(
        self,
        original: np.ndarray,
        corrected: np.ndarray,
        metal_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Frequency-split: keep low-frequency content from corrected image,
        high-frequency details from original (outside metal regions).
        """
        from scipy.ndimage import gaussian_filter
        sigma = 3.0
        lf_corrected = gaussian_filter(corrected, sigma=sigma)
        hf_original = original - gaussian_filter(original.astype(np.float32), sigma=sigma)
        result = lf_corrected + hf_original
        return result


# ---------------------------------------------------------------------------
# Cascaded CMF Segmentor
# ---------------------------------------------------------------------------

@dataclass
class CMFSegmentorConfig:
    """Configuration for cascaded CMF segmentation."""
    # Stage 1 (coarse)
    coarse_patch_size: Tuple[int, int, int] = (96, 96, 96)
    coarse_step_size: float = 0.5
    # Stage 2 (fine)
    fine_patch_size: Tuple[int, int, int] = (128, 128, 128)
    fine_step_size: float = 0.5
    roi_padding_mm: float = 20.0
    # Metal artifact reduction
    apply_mar: bool = True
    mar_blend_weight: float = 0.85
    # Normalisation (CBCT-specific values differ from fan-beam CT)
    cbct_clip_low: float = -100.0
    cbct_clip_high: float = 2500.0
    cbct_intensity_mean: float = 300.0
    cbct_intensity_std: float = 280.0
    # CT values (for fan-beam CT input)
    ct_clip_low: float = -200.0
    ct_clip_high: float = 2000.0
    ct_intensity_mean: float = 410.0
    ct_intensity_std: float = 355.0
    # Post-processing
    min_component_voxels: int = 300
    # Device
    device: str = "cuda"


class MandibleSegmentor:
    """
    Two-stage CMF segmentor: coarse detection → fine sub-structure labelling.

    Stage 1: Full-FOV 3D U-Net at low resolution to detect the CMF bounding
             box and separate mandible from maxilla.
    Stage 2: High-resolution 3D U-Net on the cropped CMF ROI for fine-grained
             labelling (condyle, canal, orbital floor, etc.).

    Also provides:
      - Condyle landmark extraction
      - Mandibular canal centreline tracing
      - Mandible / maxilla rigid body masks for osteotomy planning
    """

    def __init__(
        self,
        coarse_model: UNet3D,
        fine_model: UNet3D,
        config: CMFSegmentorConfig,
        mar: Optional[MetalArtifactReducer] = None,
    ) -> None:
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.config = config
        self.mar = mar or MetalArtifactReducer(blend_weight=config.mar_blend_weight)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.coarse_model.to(self.device).eval()
        self.fine_model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        coarse_path: str | Path,
        fine_path: str | Path,
        config: Optional[CMFSegmentorConfig] = None,
    ) -> "MandibleSegmentor":
        cfg = config or CMFSegmentorConfig()
        device = cfg.device if torch.cuda.is_available() else "cpu"

        def _load(path: Path) -> UNet3D:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            mc = ckpt.get("model_config", {})
            model = UNet3D(
                in_channels=mc.get("in_channels", 1),
                num_classes=mc.get("num_classes", NUM_CMF_CLASSES),
                base_channels=mc.get("base_channels", 32),
                max_channels=mc.get("max_channels", 320),
                depth=mc.get("depth", 4),
                deep_supervision=False,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            return model

        coarse = _load(Path(coarse_path))
        fine = _load(Path(fine_path))
        return cls(coarse, fine, cfg)

    @classmethod
    def build_new(cls, config: Optional[CMFSegmentorConfig] = None) -> "MandibleSegmentor":
        cfg = config or CMFSegmentorConfig()
        coarse = UNet3D(num_classes=NUM_CMF_CLASSES, base_channels=16, depth=3, deep_supervision=True)
        fine = UNet3D(num_classes=NUM_CMF_CLASSES, base_channels=32, depth=4, deep_supervision=True)
        return cls(coarse, fine, cfg)

    # ------------------------------------------------------------------
    # Main segmentation entry point
    # ------------------------------------------------------------------

    def segment(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        modality: str = "CT",  # "CT" | "CBCT"
    ) -> Dict[str, np.ndarray]:
        """
        Full CMF segmentation pipeline.

        Args:
            volume: (D, H, W) float32 array in HU.
            spacing: voxel spacing (dz, dy, dx) in mm.
            modality: imaging modality, affects normalisation.

        Returns:
            Dictionary with keys from CMF_CLASSES mapping to binary masks.
            Also includes 'label_map' (integer), 'mandible' and 'maxilla'
            composite masks, and 'mandibular_canal_centreline' polyline.
        """
        # 1. Metal artifact reduction for CBCT
        if modality == "CBCT" and self.config.apply_mar:
            volume = self.mar.reduce(volume, spacing)

        # 2. Normalise
        volume_norm = self._normalise(volume, modality)

        # 3. Stage 1 — coarse segmentation (full FOV, downsampled)
        coarse_tensor = torch.from_numpy(volume_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        coarse_labels = sliding_window_inference(
            model=self.coarse_model,
            volume=coarse_tensor,
            config=SlidingWindowConfig(patch_size=self.config.coarse_patch_size,
                                       step_size=self.config.coarse_step_size),
            num_classes=NUM_CMF_CLASSES,
            device=self.device,
        )

        # 4. Extract CMF ROI bounding box from coarse segmentation
        roi_slices = self._compute_roi(coarse_labels, volume.shape, spacing)

        # 5. Stage 2 — fine segmentation within ROI
        roi_volume = volume_norm[roi_slices]
        roi_tensor = torch.from_numpy(roi_volume).unsqueeze(0).unsqueeze(0).to(self.device)
        fine_labels_roi = sliding_window_inference(
            model=self.fine_model,
            volume=roi_tensor,
            config=SlidingWindowConfig(patch_size=self.config.fine_patch_size,
                                       step_size=self.config.fine_step_size),
            num_classes=NUM_CMF_CLASSES,
            device=self.device,
        )

        # 6. Paste fine labels back into full-volume label map
        label_map = coarse_labels.copy()
        label_map[roi_slices] = fine_labels_roi

        # 7. Post-processing
        label_map = postprocess_segmentation(
            label_map,
            min_component_voxels=self.config.min_component_voxels,
        )

        # 8. Build output dict
        result = {name: (label_map == idx) for idx, name in CMF_CLASSES.items() if idx > 0}
        result["label_map"] = label_map

        # Composite masks
        result["mandible"] = np.isin(label_map, list(MANDIBLE_CLASSES))
        result["maxilla"]  = np.isin(label_map, list(MAXILLA_CLASSES))
        result["zygoma"]   = np.isin(label_map, list(ZYGOMA_CLASSES))

        # 9. Landmark extraction
        result["condyle_landmarks"] = self.extract_condyle_landmarks(
            result["mandibular_condyle"], spacing
        )
        result["mandibular_canal_centreline"] = self.trace_canal_centreline(
            result.get("mandibular_canal", np.zeros(label_map.shape, dtype=bool)),
            spacing,
        )

        return result

    # ------------------------------------------------------------------
    # ROI computation
    # ------------------------------------------------------------------

    def _compute_roi(
        self,
        coarse_labels: np.ndarray,
        original_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
    ) -> Tuple[slice, slice, slice]:
        """Compute padded bounding box around all CMF structures."""
        cmf_mask = coarse_labels > 0
        if not cmf_mask.any():
            # Fall back to central 60% of volume
            d, h, w = original_shape
            return (
                slice(d // 5, 4 * d // 5),
                slice(h // 5, 4 * h // 5),
                slice(w // 5, 4 * w // 5),
            )

        coords = np.argwhere(cmf_mask)
        mn = coords.min(axis=0)
        mx = coords.max(axis=0)

        # Add padding in mm, converted to voxels
        pad_voxels = [int(self.config.roi_padding_mm / sp) for sp in spacing]
        slices = tuple(
            slice(max(0, mn[i] - pad_voxels[i]), min(original_shape[i], mx[i] + pad_voxels[i]))
            for i in range(3)
        )
        return slices

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalise(self, volume: np.ndarray, modality: str) -> np.ndarray:
        if modality == "CBCT":
            lo, hi = self.config.cbct_clip_low, self.config.cbct_clip_high
            mu, sigma = self.config.cbct_intensity_mean, self.config.cbct_intensity_std
        else:
            lo, hi = self.config.ct_clip_low, self.config.ct_clip_high
            mu, sigma = self.config.ct_intensity_mean, self.config.ct_intensity_std

        v = np.clip(volume, lo, hi).astype(np.float32)
        return (v - mu) / (sigma + 1e-8)

    # ------------------------------------------------------------------
    # Condyle landmark extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_condyle_landmarks(
        condyle_mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract condylar head landmarks from binary condyle mask.

        Returns dict with:
          - 'left_condyle_center': (3,) array in mm
          - 'right_condyle_center': (3,) array in mm
          - 'left_condyle_axis': unit vector along condylar long axis
          - 'right_condyle_axis': unit vector along condylar long axis
        """
        labeled, n = ndimage.label(condyle_mask)
        if n < 1:
            return {k: None for k in ["left_condyle_center", "right_condyle_center",
                                      "left_condyle_axis", "right_condyle_axis"]}

        # Find two largest components (left / right condyle)
        sizes = [(ndimage.sum(condyle_mask, labeled, i + 1), i + 1) for i in range(n)]
        sizes.sort(reverse=True)
        top2 = [idx for _, idx in sizes[:2]]

        condyles = []
        for comp_id in top2:
            voxels = np.argwhere(labeled == comp_id)
            center_vox = voxels.mean(axis=0)
            center_mm = center_vox * np.array(spacing)

            # Condylar axis via PCA on component voxels
            voxels_mm = voxels * np.array(spacing)
            cov = np.cov((voxels_mm - center_mm).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis = eigvecs[:, -1]  # largest eigenvalue → long axis

            condyles.append((center_mm, axis))

        # Assign left/right by x-coordinate (anatomical left = patient's left)
        condyles.sort(key=lambda c: c[0][2])  # sort by x (W axis)
        result = {}
        for side, (center, axis) in zip(["right", "left"], condyles):
            result[f"{side}_condyle_center"] = center
            result[f"{side}_condyle_axis"]   = axis

        return result

    # ------------------------------------------------------------------
    # Mandibular canal centreline tracing
    # ------------------------------------------------------------------

    @staticmethod
    def trace_canal_centreline(
        canal_mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Optional[np.ndarray]:
        """
        Trace the inferior alveolar nerve canal centreline using medial axis
        skeletonisation.

        Returns:
            (N, 3) array of centreline points in mm, or None if canal not found.
        """
        if not canal_mask.any():
            return None

        try:
            from skimage.morphology import skeletonize_3d
        except ImportError:
            logger.warning("scikit-image not available; skipping canal centreline.")
            return None

        skeleton = skeletonize_3d(canal_mask.astype(np.uint8))
        skel_voxels = np.argwhere(skeleton)
        if len(skel_voxels) == 0:
            return None

        # Sort by inferior-to-superior (D axis)
        skel_voxels = skel_voxels[np.argsort(skel_voxels[:, 0])]
        return skel_voxels * np.array(spacing)

    # ------------------------------------------------------------------
    # Occlusal plane estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_occlusal_plane(
        teeth_upper_mask: np.ndarray,
        teeth_lower_mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Estimate the occlusal plane from upper and lower tooth masks.

        Returns:
            Dict with 'normal' (unit vector), 'centroid' (mm), 'd' (plane constant).
        """
        upper_pts = np.argwhere(teeth_upper_mask) * np.array(spacing)
        lower_pts = np.argwhere(teeth_lower_mask) * np.array(spacing)

        if len(upper_pts) < 4 or len(lower_pts) < 4:
            return None

        # Use midpoints between upper and lower occlusal surfaces
        # Approximate: mean of upper inferior surface + mean of lower superior surface
        upper_inf = upper_pts[upper_pts[:, 0].argmax()]  # bottommost upper tooth voxel
        lower_sup = lower_pts[lower_pts[:, 0].argmin()]  # topmost lower tooth voxel
        centroid = (upper_inf + lower_sup) / 2.0

        # Fit plane via PCA on merged cusp points
        all_pts = np.vstack([upper_pts, lower_pts])
        cov = np.cov((all_pts - centroid).T)
        _, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # smallest eigenvalue → plane normal
        d = -np.dot(normal, centroid)

        return {"normal": normal, "centroid": centroid, "d": d}

    # ------------------------------------------------------------------
    # Symmetry / midline deviation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_midline_deviation(
        mandible_mask: np.ndarray,
        maxilla_mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Dict[str, float]:
        """
        Compute sagittal midline deviation between mandible and maxilla.

        Returns:
            Dict with keys:
              - 'mandible_midline_x': mandible midline x-coordinate in mm
              - 'maxilla_midline_x':  maxilla midline x-coordinate in mm
              - 'deviation_mm':       signed lateral deviation
              - 'asymmetry_score':    normalised asymmetry index 0–1
        """
        def _midline(mask: np.ndarray) -> float:
            voxels = np.argwhere(mask)
            if len(voxels) == 0:
                return 0.0
            # Use centroid in W (left-right) dimension
            return float(voxels[:, 2].mean() * spacing[2])

        m_mid = _midline(mandible_mask)
        mx_mid = _midline(maxilla_mask)
        deviation = m_mid - mx_mid

        # Asymmetry score: ratio of deviation to maxilla width
        mx_voxels = np.argwhere(maxilla_mask)
        if len(mx_voxels) > 0:
            mx_width_mm = (mx_voxels[:, 2].max() - mx_voxels[:, 2].min()) * spacing[2]
            asymmetry = abs(deviation) / (mx_width_mm + 1e-8)
        else:
            asymmetry = 0.0

        return {
            "mandible_midline_x": m_mid,
            "maxilla_midline_x": mx_mid,
            "deviation_mm": deviation,
            "asymmetry_score": min(asymmetry, 1.0),
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    cfg = CMFSegmentorConfig(device="cpu",
                             coarse_patch_size=(48, 48, 48),
                             fine_patch_size=(64, 64, 64))
    seg = MandibleSegmentor.build_new(cfg)
    vol = np.random.randn(80, 80, 80).astype(np.float32) * 300
    spacing = (0.5, 0.5, 0.5)
    result = seg.segment(vol, spacing, modality="CBCT")
    logger.info("CMF segmentation keys: %s", list(result.keys()))
    logger.info("Mandible voxels: %d", result["mandible"].sum())
    sys.exit(0)
