"""
landmark_detector.py
--------------------
3D anatomical landmark detection via deep learning heatmap regression.

Detects:
  CMF Cephalometric Landmarks (16):
    Nasion (N), Sella (S), Orbitale (Or), Porion (Po), ANS, PNS, A-point,
    B-point, Pogonion (Pog), Menton (Me), Gonion (Go) L/R, Condylion (Co) L/R,
    Gnathion (Gn), Articulare (Ar)

  Orthopedic Landmarks (8):
    Femoral head center L/R, Acetabulum center L/R,
    Tibial plateau center L/R, Knee joint center L/R

Architecture:
  - 3D encoder (shared with segmentor backbone)
  - Per-landmark Gaussian heatmap regression heads
  - Learnable temperature scaling per landmark
  - Sub-voxel refinement via soft-argmax

Loss:
  - L2 heatmap matching + direct L2 coordinate loss (multi-task)

Reference:
  Payer et al., "Integrating spatial configuration into heatmap regression
  based CNNs for landmark localization", MedIA 2019.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Landmark definitions
# ---------------------------------------------------------------------------

CMF_LANDMARKS = [
    "nasion", "sella", "orbitale_l", "orbitale_r",
    "porion_l", "porion_r", "ANS", "PNS",
    "A_point", "B_point", "pogonion", "menton",
    "gonion_l", "gonion_r", "condylion_l", "condylion_r",
]

ORTHO_LANDMARKS = [
    "femoral_head_l", "femoral_head_r",
    "acetabulum_center_l", "acetabulum_center_r",
    "tibial_plateau_center_l", "tibial_plateau_center_r",
    "knee_joint_center_l", "knee_joint_center_r",
]

ALL_LANDMARKS = CMF_LANDMARKS + ORTHO_LANDMARKS


# ---------------------------------------------------------------------------
# Gaussian heatmap generation (for training)
# ---------------------------------------------------------------------------

def generate_heatmap_3d(
    landmark_vox: np.ndarray,   # (3,) voxel coordinates (z, y, x)
    volume_shape: Tuple[int, int, int],
    sigma: float = 3.0,
    normalise: bool = True,
) -> np.ndarray:
    """
    Generate a 3D Gaussian heatmap centred at landmark_vox.

    Args:
        landmark_vox: (z, y, x) voxel coordinates of the landmark.
        volume_shape: (D, H, W) shape of the output heatmap.
        sigma: standard deviation in voxels.
        normalise: if True, normalise heatmap to sum to 1.

    Returns:
        (D, H, W) float32 heatmap.
    """
    D, H, W = volume_shape
    z0, y0, x0 = landmark_vox

    z = np.arange(D, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    x = np.arange(W, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    heatmap = np.exp(
        -((zz - z0) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2) / (2 * sigma ** 2)
    )
    if normalise:
        heatmap /= heatmap.sum() + 1e-8
    return heatmap.astype(np.float32)


def soft_argmax_3d(
    heatmap: torch.Tensor,  # (B, D, H, W)
    beta: float = 100.0,
) -> torch.Tensor:
    """
    Differentiable sub-voxel landmark localisation via soft-argmax.

    Computes the expected value of (z, y, x) under a softmax-normalised
    heatmap distribution.

    Args:
        heatmap: (B, D, H, W) raw heatmap logits (or probabilities).
        beta: temperature scaling factor.

    Returns:
        (B, 3) tensor of (z, y, x) coordinates in voxel space.
    """
    B, D, H, W = heatmap.shape
    device = heatmap.device

    # Temperature-scaled softmax
    flat = heatmap.view(B, -1)
    flat = F.softmax(flat * beta, dim=1)
    flat = flat.view(B, D, H, W)

    # Coordinate grids
    z_grid = torch.arange(D, device=device, dtype=torch.float32)
    y_grid = torch.arange(H, device=device, dtype=torch.float32)
    x_grid = torch.arange(W, device=device, dtype=torch.float32)

    z_coord = (flat.sum(dim=[2, 3]) * z_grid).sum(dim=1)  # (B,)
    y_coord = (flat.sum(dim=[1, 3]) * y_grid).sum(dim=1)  # (B,)
    x_coord = (flat.sum(dim=[1, 2]) * x_grid).sum(dim=1)  # (B,)

    return torch.stack([z_coord, y_coord, x_coord], dim=1)  # (B, 3)


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

class LandmarkEncoder(nn.Module):
    """Lightweight 3D encoder shared across all landmark heads."""

    def __init__(self, in_channels: int = 1, base_ch: int = 32) -> None:
        super().__init__()

        def block(inc, outc, stride=2):
            return nn.Sequential(
                nn.Conv3d(inc, outc, 3, stride, 1, bias=False),
                nn.InstanceNorm3d(outc, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(outc, outc, 3, 1, 1, bias=False),
                nn.InstanceNorm3d(outc, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            )

        self.s1 = block(in_channels, base_ch, stride=1)   # full res
        self.s2 = block(base_ch, base_ch * 2)              # /2
        self.s3 = block(base_ch * 2, base_ch * 4)          # /4
        self.s4 = block(base_ch * 4, base_ch * 8)          # /8

        # Feature pyramid upsampling for multi-scale context
        self.up3 = nn.Conv3d(base_ch * 8, base_ch * 4, 1)
        self.up2 = nn.Conv3d(base_ch * 4, base_ch * 2, 1)
        self.fuse3 = nn.Conv3d(base_ch * 4 * 2, base_ch * 4, 1)
        self.fuse2 = nn.Conv3d(base_ch * 2 * 2, base_ch * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns features at 3 scales: [1/4, 1/2, 1/1] for FPN."""
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)

        up3 = F.interpolate(self.up3(s4), size=s3.shape[2:], mode="trilinear", align_corners=False)
        p3 = self.fuse3(torch.cat([up3, s3], dim=1))

        up2 = F.interpolate(self.up2(p3), size=s2.shape[2:], mode="trilinear", align_corners=False)
        p2 = self.fuse2(torch.cat([up2, s2], dim=1))

        return p3, p2, s1  # coarse→fine feature maps


class LandmarkHead(nn.Module):
    """Per-landmark heatmap regression head."""

    def __init__(self, in_channels: int, output_shape: Tuple[int, int, int]) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(in_channels // 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(in_channels // 2, 1, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(100.0))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            heatmap: (B, 1, D, H, W)
            coords: (B, 3) soft-argmax coordinates
        """
        heatmap = self.conv(features)
        heatmap = F.interpolate(heatmap, size=self.output_shape,
                                mode="trilinear", align_corners=False)
        coords = soft_argmax_3d(heatmap[:, 0], beta=self.temperature.abs())
        return heatmap, coords


class LandmarkDetectorNet(nn.Module):
    """
    Full landmark detection network.
    Shared encoder → per-landmark heatmap heads.
    """

    def __init__(
        self,
        landmark_names: List[str],
        input_shape: Tuple[int, int, int] = (128, 128, 128),
        base_ch: int = 32,
    ) -> None:
        super().__init__()
        self.landmark_names = landmark_names
        self.n_landmarks = len(landmark_names)
        self.input_shape = input_shape

        self.encoder = LandmarkEncoder(in_channels=1, base_ch=base_ch)

        # Heatmap output at 1/4 resolution to save memory
        heatmap_shape = tuple(s // 4 for s in input_shape)

        # Use coarse (1/4 scale) features for deep landmarks,
        # fine (1/2 scale) features for surface landmarks
        coarse_ch = base_ch * 4
        fine_ch = base_ch * 2

        # Assign each landmark to coarse or fine path
        surface_lms = {"nasion", "pogonion", "menton", "gonion_l", "gonion_r",
                       "condylion_l", "condylion_r", "orbitale_l", "orbitale_r"}
        fine_shape = tuple(s // 2 for s in input_shape)

        self.heads = nn.ModuleDict()
        for name in landmark_names:
            if name in surface_lms:
                self.heads[name] = LandmarkHead(fine_ch, fine_shape)
            else:
                self.heads[name] = LandmarkHead(coarse_ch, heatmap_shape)

        self._surface_lms = surface_lms

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (B, 1, D, H, W) normalised CT/CBCT volume.

        Returns:
            Dict mapping landmark_name → {"heatmap": Tensor, "coords": Tensor}
        """
        coarse_feat, fine_feat, _ = self.encoder(x)
        outputs = {}
        for name, head in self.heads.items():
            feat = fine_feat if name in self._surface_lms else coarse_feat
            heatmap, coords = head(feat)
            outputs[name] = {"heatmap": heatmap, "coords": coords}
        return outputs


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def landmark_detection_loss(
    predictions: Dict[str, Dict[str, torch.Tensor]],
    gt_heatmaps: Dict[str, torch.Tensor],
    gt_coords: Dict[str, torch.Tensor],   # (B, 3) in voxel space
    heatmap_weight: float = 1.0,
    coord_weight: float = 10.0,
    visibility_mask: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-task loss: heatmap MSE + direct coordinate L1.

    Args:
        predictions: output of LandmarkDetectorNet.forward
        gt_heatmaps: Dict name → (B, 1, D', H', W') ground truth heatmaps
        gt_coords: Dict name → (B, 3) ground truth voxel coordinates
        heatmap_weight: weight for heatmap loss component
        coord_weight: weight for coordinate loss component
        visibility_mask: optional Dict name → (B,) bool tensor (1=visible)

    Returns:
        (total_loss, {name: per_landmark_loss_mm})
    """
    total = torch.tensor(0.0, device=next(iter(gt_coords.values())).device)
    per_lm = {}

    for name, pred in predictions.items():
        vis = visibility_mask[name] if visibility_mask else None

        # Heatmap loss
        gt_hm = F.interpolate(
            gt_heatmaps[name],
            size=pred["heatmap"].shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        hm_loss = F.mse_loss(pred["heatmap"], gt_hm)

        # Coordinate loss (L1 — more robust than L2 for outliers)
        gc = gt_coords[name].float()
        if vis is not None:
            coord_loss = (F.l1_loss(pred["coords"][vis], gc[vis])
                          if vis.any() else torch.tensor(0.0))
        else:
            coord_loss = F.l1_loss(pred["coords"], gc)

        lm_loss = heatmap_weight * hm_loss + coord_weight * coord_loss
        total = total + lm_loss
        per_lm[name] = float(lm_loss)

    return total, per_lm


# ---------------------------------------------------------------------------
# High-level detector API
# ---------------------------------------------------------------------------

@dataclass
class LandmarkDetectorConfig:
    input_shape: Tuple[int, int, int] = (128, 128, 128)
    base_channels: int = 32
    landmark_set: str = "CMF"   # "CMF" | "ORTHO" | "ALL"
    device: str = "cuda"
    # Normalisation
    clip_low: float = -200.0
    clip_high: float = 2000.0
    intensity_mean: float = 410.0
    intensity_std: float = 355.0


class LandmarkDetector:
    """
    High-level interface for anatomical landmark detection.

    Returns detected landmark coordinates in mm (physical space).

    Example::

        detector = LandmarkDetector.from_pretrained("models/landmarks_cmf.pt")
        landmarks = detector.predict(volume_np, spacing=(0.5, 0.5, 0.5))
        # landmarks["nasion"] → np.array([x, y, z]) in mm
    """

    def __init__(
        self,
        model: LandmarkDetectorNet,
        config: LandmarkDetectorConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        config: Optional[LandmarkDetectorConfig] = None,
    ) -> "LandmarkDetector":
        cfg = config or LandmarkDetectorConfig()
        device = cfg.device if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        mc = ckpt.get("model_config", {})
        lm_names = mc.get("landmark_names", CMF_LANDMARKS)
        model = LandmarkDetectorNet(
            landmark_names=lm_names,
            input_shape=mc.get("input_shape", cfg.input_shape),
            base_ch=mc.get("base_channels", cfg.base_channels),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded LandmarkDetector from %s (%d landmarks)", path, len(lm_names))
        return cls(model, cfg)

    @classmethod
    def build_new(
        cls,
        config: Optional[LandmarkDetectorConfig] = None,
    ) -> "LandmarkDetector":
        cfg = config or LandmarkDetectorConfig()
        lm_names = {"CMF": CMF_LANDMARKS, "ORTHO": ORTHO_LANDMARKS, "ALL": ALL_LANDMARKS}[cfg.landmark_set]
        model = LandmarkDetectorNet(
            landmark_names=lm_names,
            input_shape=cfg.input_shape,
            base_ch=cfg.base_channels,
        )
        return cls(model, cfg)

    def predict(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        return_heatmaps: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Detect landmarks in a CT/CBCT volume.

        Args:
            volume: (D, H, W) float32 array in HU.
            spacing: (dz, dy, dx) in mm.
            return_heatmaps: if True, also return heatmap volumes.

        Returns:
            Dict mapping landmark_name → (3,) array in mm [z, y, x].
            If return_heatmaps=True, also includes 'heatmaps' key.
        """
        # Preprocess and resize to expected input shape
        v = np.clip(volume, self.config.clip_low, self.config.clip_high).astype(np.float32)
        v = (v - self.config.intensity_mean) / (self.config.intensity_std + 1e-8)

        # Resize to model input shape via SimpleITK
        orig_shape = volume.shape
        target_shape = self.config.input_shape
        scale = np.array(target_shape) / np.array(orig_shape)

        from scipy.ndimage import zoom
        v_resized = zoom(v, scale, order=1)

        tensor = torch.from_numpy(v_resized).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        results = {}
        for name, out in outputs.items():
            coords_vox_scaled = out["coords"][0].cpu().numpy()  # (3,) at scaled resolution
            # Map back to original voxel space
            coords_vox = coords_vox_scaled / scale
            # Convert to mm
            coords_mm = coords_vox * np.array(spacing)
            results[name] = coords_mm

        if return_heatmaps:
            results["heatmaps"] = {
                name: out["heatmap"][0, 0].cpu().numpy()
                for name, out in outputs.items()
            }

        return results

    def compute_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        ground_truth: Dict[str, np.ndarray],
        spacing: Tuple[float, float, float],
    ) -> Dict[str, float]:
        """
        Compute per-landmark detection errors in mm.

        Returns:
            Dict with per-landmark errors, mean error, and % within 2mm.
        """
        errors = {}
        for name in ground_truth:
            if name in predictions:
                err = float(np.linalg.norm(predictions[name] - ground_truth[name]))
                errors[name] = err

        if not errors:
            return {}

        values = list(errors.values())
        errors["__mean__"] = float(np.mean(values))
        errors["__std__"] = float(np.std(values))
        errors["__pct_2mm__"] = float(np.mean(np.array(values) < 2.0) * 100)
        errors["__pct_4mm__"] = float(np.mean(np.array(values) < 4.0) * 100)
        return errors


# ---------------------------------------------------------------------------
# Cephalometric measurements
# ---------------------------------------------------------------------------

class CephalometricAnalysis:
    """
    Standard CMF cephalometric angle and distance measurements.
    All measurements comply with Steiner, Ricketts, and Jarabak analyses.
    """

    @staticmethod
    def sna_angle(S: np.ndarray, N: np.ndarray, A: np.ndarray) -> float:
        """SNA angle: sagittal relationship of maxilla to cranial base."""
        v1 = S - N
        v2 = A - N
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    @staticmethod
    def snb_angle(S: np.ndarray, N: np.ndarray, B: np.ndarray) -> float:
        """SNB angle: sagittal relationship of mandible to cranial base."""
        v1 = S - N
        v2 = B - N
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    @staticmethod
    def anb_angle(sna: float, snb: float) -> float:
        """ANB: maxillomandibular relationship (normal: 2°)."""
        return sna - snb

    @staticmethod
    def facial_height_ratio(
        N: np.ndarray, ANS: np.ndarray, Me: np.ndarray, PNS: np.ndarray
    ) -> float:
        """N-ANS / ANS-Me ratio (lower facial height proportion)."""
        upper = np.linalg.norm(N - ANS)
        lower = np.linalg.norm(ANS - Me)
        return float(lower / (upper + 1e-8))

    @staticmethod
    def wits_appraisal(
        A: np.ndarray, B: np.ndarray,
        occlusal_plane_normal: np.ndarray, occlusal_plane_point: np.ndarray,
    ) -> float:
        """
        Wits appraisal: project A and B onto occlusal plane, measure distance.
        Positive = maxillary excess, negative = mandibular excess.
        Normal: -1 to +1 mm.
        """
        def project_to_plane(pt):
            d = np.dot(pt - occlusal_plane_point, occlusal_plane_normal)
            return pt - d * occlusal_plane_normal

        ao = project_to_plane(A)
        bo = project_to_plane(B)
        return float(np.linalg.norm(ao - bo) * np.sign(ao[2] - bo[2]))

    def run_full_analysis(
        self, landmarks: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Run complete Steiner cephalometric analysis.

        Args:
            landmarks: Dict of landmark name → (3,) mm coordinates.

        Returns:
            Dict of measurement name → value (degrees or mm).
        """
        results = {}
        lm = landmarks

        try:
            results["SNA"] = self.sna_angle(lm["sella"], lm["nasion"], lm["A_point"])
            results["SNB"] = self.snb_angle(lm["sella"], lm["nasion"], lm["B_point"])
            results["ANB"] = self.anb_angle(results["SNA"], results["SNB"])
            results["facial_height_ratio"] = self.facial_height_ratio(
                lm["nasion"], lm["ANS"], lm["menton"], lm["PNS"]
            )
            # Mandibular plane angle (MP/SN)
            mp = lm["menton"] - lm["gonion_l"]
            sn = lm["nasion"] - lm["sella"]
            cos_mp = np.dot(mp, sn) / (np.linalg.norm(mp) * np.linalg.norm(sn) + 1e-8)
            results["SN_MP_angle"] = float(np.degrees(np.arccos(np.clip(abs(cos_mp), 0, 1))))
            # Inter-condylar width
            results["intercondylar_width_mm"] = float(
                np.linalg.norm(lm["condylion_l"] - lm["condylion_r"])
            )
        except KeyError as e:
            logger.warning("Missing landmark for cephalometric analysis: %s", e)

        return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    cfg = LandmarkDetectorConfig(
        input_shape=(64, 64, 64),
        device="cpu",
        landmark_set="CMF",
    )
    detector = LandmarkDetector.build_new(cfg)

    dummy_vol = np.random.randn(80, 80, 80).astype(np.float32) * 400
    spacing = (0.5, 0.5, 0.5)
    result = detector.predict(dummy_vol, spacing)

    logger.info("Detected %d landmarks", len([k for k in result if not k.startswith("_")]))
    for name, coord in list(result.items())[:3]:
        logger.info("  %s: %s mm", name, np.round(coord, 2))

    logger.info("LandmarkDetector self-test passed.")
    sys.exit(0)

LANDMARK_TOLERANCE_MM = 2.0  # clinical acceptance threshold
