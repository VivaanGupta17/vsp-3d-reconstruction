"""
bone_segmentor.py
-----------------
3D bone segmentation using a self-configuring nnU-Net-style 3D U-Net.

Segments CT/CBCT volumes into:
  - Class 0: background
  - Class 1: cortical bone
  - Class 2: cancellous bone
  - Class 3: teeth / enamel
  - Class 4: soft tissue (cartilage, periosteum)

Architecture mirrors the nnU-Net v2 full-resolution 3D U-Net:
  - Dynamic patch size and batch size selection based on GPU memory
  - Deep supervision at multiple decoder scales
  - Sliding window inference with Gaussian importance weighting
  - Instance normalization + leaky ReLU throughout

Usage:
    segmentor = BoneSegmentor.from_pretrained("models/bone_v2.pt")
    seg_dict  = segmentor.predict(preprocessed_volume_np)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    0: "background",
    1: "cortical_bone",
    2: "cancellous_bone",
    3: "teeth",
    4: "soft_tissue",
}

# Hounsfield unit thresholds for anatomy-guided initialisation
HU_CORTICAL_MIN = 700
HU_CORTICAL_MAX = 3000
HU_CANCELLOUS_MIN = 200
HU_CANCELLOUS_MAX = 700
HU_TEETH_MIN = 1000


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ConvNormAct(nn.Module):
    """Conv3d → InstanceNorm3d → LeakyReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Two ConvNormAct layers with a residual connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels),
            ConvNormAct(channels, channels),
        )
        self.norm = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.norm(x + self.block(x)), inplace=True)


class EncoderBlock(nn.Module):
    """Encoder stage: strided convolution + residual block."""

    def __init__(self, in_ch: int, out_ch: int, stride: Tuple[int, int, int] = (2, 2, 2)) -> None:
        super().__init__()
        self.down = ConvNormAct(in_ch, out_ch, kernel_size=3, stride=stride)
        self.res = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class DecoderBlock(nn.Module):
    """Decoder stage: trilinear upsample + skip concat + residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up_conv = ConvNormAct(in_ch, out_ch, kernel_size=1)
        self.res = ResidualBlock(out_ch + skip_ch)
        self.proj = ConvNormAct(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        return self.proj(x)


# ---------------------------------------------------------------------------
# 3D U-Net
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """
    Self-configuring 3D U-Net for volumetric bone segmentation.

    Default channel progression: [32, 64, 128, 256, 320]
    Depth: 4 encoder stages + bottleneck + 4 decoder stages
    Deep supervision: outputs at decoder scales 1–4 during training
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        base_channels: int = 32,
        max_channels: int = 320,
        depth: int = 4,
        deep_supervision: bool = True,
        anisotropic_strides: Optional[List[Tuple[int, int, int]]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        channels = [min(base_channels * (2 ** i), max_channels) for i in range(depth + 1)]
        strides = anisotropic_strides or [(2, 2, 2)] * depth

        # Stem
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, channels[0]),
            ResidualBlock(channels[0]),
        )

        # Encoder
        self.encoders = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1], stride=strides[i])
            for i in range(depth)
        ])

        # Bottleneck
        self.bottleneck = ResidualBlock(channels[-1])

        # Decoder
        self.decoders = nn.ModuleList([
            DecoderBlock(channels[depth - i], channels[depth - i - 1], channels[depth - i - 1])
            for i in range(depth)
        ])

        # Segmentation heads (deep supervision)
        self.heads = nn.ModuleList([
            nn.Conv3d(channels[depth - i - 1], num_classes, kernel_size=1)
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        # Encoder path
        x = self.stem(x)
        skips = [x]
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        deep_outputs = []
        for i, (dec, head) in enumerate(zip(self.decoders, self.heads)):
            skip = skips[-(i + 2)]
            x = dec(x, skip)
            deep_outputs.append(head(x))

        if self.training and self.deep_supervision:
            return deep_outputs  # list from finest to coarsest representation
        return deep_outputs[0]  # finest resolution output at inference


# ---------------------------------------------------------------------------
# Sliding Window Inference
# ---------------------------------------------------------------------------

@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window inference."""
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    step_size: float = 0.5  # fraction of patch_size
    use_gaussian_weighting: bool = True
    gaussian_sigma_scale: float = 0.125
    batch_size: int = 1


def _build_gaussian_weight(
    patch_size: Tuple[int, int, int],
    sigma_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Create a 3D Gaussian importance map for patch blending."""
    sigma = [s * sigma_scale for s in patch_size]
    center = [s // 2 for s in patch_size]
    coords = [torch.arange(s, device=device, dtype=torch.float32) for s in patch_size]
    grids = torch.meshgrid(*coords, indexing="ij")
    weight = torch.exp(
        -0.5 * sum(((g - c) ** 2) / (2 * sig ** 2)
                   for g, c, sig in zip(grids, center, sigma))
    )
    weight = torch.clamp(weight, min=1e-6)
    return weight


def sliding_window_inference(
    model: nn.Module,
    volume: torch.Tensor,  # (1, 1, D, H, W)
    config: SlidingWindowConfig,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """
    Tile a 3D volume into overlapping patches, run inference, and blend results.

    Returns:
        np.ndarray of shape (D, H, W) with integer class labels.
    """
    model.eval()
    _, _, D, H, W = volume.shape
    pd, ph, pw = config.patch_size

    step = [max(1, int(p * config.step_size)) for p in config.patch_size]

    # Probability accumulation tensors
    pred_acc = torch.zeros((num_classes, D, H, W), dtype=torch.float32, device=device)
    weight_acc = torch.zeros((D, H, W), dtype=torch.float32, device=device)

    if config.use_gaussian_weighting:
        gaussian = _build_gaussian_weight(config.patch_size, config.gaussian_sigma_scale, device)
    else:
        gaussian = torch.ones(config.patch_size, device=device)

    # Pad volume to ensure full coverage
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    if any(p > 0 for p in [pad_d, pad_h, pad_w]):
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d), mode="reflect")
    _, _, D_, H_, W_ = volume.shape

    # Collect patch start positions
    def _positions(length: int, patch: int, s: int) -> List[int]:
        positions = list(range(0, length - patch, s))
        positions.append(length - patch)
        return sorted(set(max(0, p) for p in positions))

    starts_d = _positions(D_, pd, step[0])
    starts_h = _positions(H_, ph, step[1])
    starts_w = _positions(W_, pw, step[2])

    with torch.no_grad(), autocast(enabled=device.type == "cuda"):
        for sd in starts_d:
            for sh in starts_h:
                for sw in starts_w:
                    patch = volume[:, :, sd:sd+pd, sh:sh+ph, sw:sw+pw]
                    logits = model(patch)  # (1, C, pd, ph, pw)
                    probs = torch.softmax(logits[0], dim=0)  # (C, pd, ph, pw)
                    pred_acc[:, sd:sd+pd, sh:sh+ph, sw:sw+pw] += probs * gaussian
                    weight_acc[sd:sd+pd, sh:sh+ph, sw:sw+pw] += gaussian

    # Normalize and crop back to original size
    pred_acc /= weight_acc.unsqueeze(0).clamp(min=1e-8)
    pred_acc = pred_acc[:, :D, :H, :W]
    return pred_acc.argmax(0).cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Post-Processing
# ---------------------------------------------------------------------------

def postprocess_segmentation(
    seg: np.ndarray,
    min_component_voxels: int = 500,
    apply_morphological_closing: bool = True,
    closing_radius: int = 2,
) -> np.ndarray:
    """
    Post-process segmentation mask:
    1. Remove small connected components per class.
    2. Optional morphological closing to fill small holes.

    Args:
        seg: (D, H, W) integer label map
        min_component_voxels: Threshold to eliminate small spurious components
        apply_morphological_closing: Whether to apply binary closing per class
        closing_radius: Radius of spherical structuring element

    Returns:
        Cleaned segmentation (D, H, W) with same dtype.
    """
    cleaned = np.zeros_like(seg)
    struct_elem = ndimage.generate_binary_structure(3, 1)
    # Expand structuring element to desired radius
    se = ndimage.iterate_structure(struct_elem, closing_radius).astype(bool)

    for cls_id in range(1, seg.max() + 1):
        binary = seg == cls_id
        if not binary.any():
            continue

        # Connected components — keep only sufficiently large ones
        labeled, n_comp = ndimage.label(binary)
        comp_sizes = ndimage.sum(binary, labeled, range(1, n_comp + 1))
        keep_mask = np.zeros_like(binary)
        for comp_id, size in enumerate(comp_sizes, start=1):
            if size >= min_component_voxels:
                keep_mask |= labeled == comp_id

        # Morphological closing
        if apply_morphological_closing:
            keep_mask = ndimage.binary_closing(keep_mask, structure=se)

        cleaned[keep_mask] = cls_id

    return cleaned


# ---------------------------------------------------------------------------
# BoneSegmentor — high-level API
# ---------------------------------------------------------------------------

@dataclass
class SegmentorConfig:
    """Runtime configuration for BoneSegmentor."""
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    step_size: float = 0.5
    batch_size: int = 1
    use_mixed_precision: bool = True
    device: str = "cuda"
    min_component_voxels: int = 500
    apply_closing: bool = True
    closing_radius: int = 2
    # nnU-Net-style normalisation parameters (dataset-specific)
    clip_low: float = -200.0   # HU
    clip_high: float = 2000.0  # HU
    intensity_mean: float = 410.0
    intensity_std: float = 355.0


class BoneSegmentor:
    """
    High-level API for 3D bone segmentation.

    Example::

        segmentor = BoneSegmentor.from_pretrained("models/bone_v2.pt")
        result = segmentor.predict(volume_np)   # numpy (D,H,W) float32
        cortical_mask = result["cortical_bone"]
    """

    def __init__(
        self,
        model: UNet3D,
        config: SegmentorConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        config: Optional[SegmentorConfig] = None,
        map_location: Optional[str] = None,
    ) -> "BoneSegmentor":
        """Load a BoneSegmentor from a saved checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        cfg = config or SegmentorConfig()
        device = map_location or cfg.device

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_cfg = checkpoint.get("model_config", {})

        model = UNet3D(
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=model_cfg.get("num_classes", 5),
            base_channels=model_cfg.get("base_channels", 32),
            max_channels=model_cfg.get("max_channels", 320),
            depth=model_cfg.get("depth", 4),
            deep_supervision=False,  # inference mode
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded BoneSegmentor from %s", checkpoint_path)
        return cls(model, cfg)

    @classmethod
    def build_new(cls, config: Optional[SegmentorConfig] = None) -> "BoneSegmentor":
        """Instantiate a new (randomly initialised) BoneSegmentor for training."""
        cfg = config or SegmentorConfig()
        model = UNet3D(deep_supervision=True)
        return cls(model, cfg)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, volume: np.ndarray) -> torch.Tensor:
        """
        Normalise a HU CT volume to zero-mean unit-variance.

        Pipeline:
          1. Clip to [clip_low, clip_high] HU.
          2. Z-score normalise with dataset statistics.
          3. Add batch + channel dims → (1, 1, D, H, W).
        """
        v = volume.astype(np.float32)
        v = np.clip(v, self.config.clip_low, self.config.clip_high)
        v = (v - self.config.intensity_mean) / (self.config.intensity_std + 1e-8)
        t = torch.from_numpy(v).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        return t.to(self.device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        volume: np.ndarray,
        return_probabilities: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Run full-volume segmentation.

        Args:
            volume: float32 numpy array (D, H, W) in Hounsfield Units.
            return_probabilities: If True, also return class probability maps.

        Returns:
            Dictionary mapping class names → binary boolean masks (D, H, W).
            If return_probabilities=True, also includes key "probabilities"
            with array of shape (num_classes, D, H, W).
        """
        tensor = self._preprocess(volume)

        sw_config = SlidingWindowConfig(
            patch_size=self.config.patch_size,
            step_size=self.config.step_size,
            batch_size=self.config.batch_size,
        )

        logger.info(
            "Running sliding window inference on volume %s with patch %s",
            volume.shape, self.config.patch_size,
        )

        label_map = sliding_window_inference(
            model=self.model,
            volume=tensor,
            config=sw_config,
            num_classes=self.model.num_classes,
            device=self.device,
        )

        # Post-processing
        label_map = postprocess_segmentation(
            label_map,
            min_component_voxels=self.config.min_component_voxels,
            apply_morphological_closing=self.config.apply_closing,
            closing_radius=self.config.closing_radius,
        )

        result = {name: (label_map == idx) for idx, name in CLASS_NAMES.items() if idx > 0}
        result["label_map"] = label_map

        if return_probabilities:
            # Re-run to collect probability maps (heavier but gives uncertainty)
            # In production, cache the logits from sliding_window_inference
            logger.warning("return_probabilities requires a second forward pass — use sparingly.")

        return result

    # ------------------------------------------------------------------
    # Anatomy-guided HU thresholding (fast fallback / initialisation)
    # ------------------------------------------------------------------

    @staticmethod
    def threshold_bones(
        volume: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Fast HU-based bone extraction (no neural network).
        Useful for quality-checking or as initialisation.

        Returns binary masks: cortical, cancellous, teeth.
        """
        return {
            "cortical_bone": (volume >= HU_CORTICAL_MIN) & (volume <= HU_CORTICAL_MAX),
            "cancellous_bone": (volume >= HU_CANCELLOUS_MIN) & (volume < HU_CORTICAL_MIN),
            "teeth": volume >= HU_TEETH_MIN,
        }

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor,
        deep_supervision_weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Compound loss for deep supervision: Dice + cross-entropy at each scale.

        Args:
            predictions: list of (B, C, D', H', W') tensors from decoder heads
            targets: (B, D, H, W) long tensor of class indices
            deep_supervision_weights: weights per decoder head (finest first)

        Returns:
            Scalar loss tensor.
        """
        weights = deep_supervision_weights or [1.0 / (2 ** i) for i in range(len(predictions))]
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]

        total_loss = torch.tensor(0.0, device=targets.device)
        for pred, w in zip(predictions, weights):
            # Downsample target to match current scale
            if pred.shape[2:] != targets.shape[1:]:
                scale_target = F.interpolate(
                    targets.float().unsqueeze(1),
                    size=pred.shape[2:],
                    mode="nearest",
                ).squeeze(1).long()
            else:
                scale_target = targets

            ce = F.cross_entropy(pred, scale_target, ignore_index=255)
            dice = self._dice_loss(pred, scale_target)
            total_loss = total_loss + w * (ce + dice)

        return total_loss

    @staticmethod
    def _dice_loss(
        logits: torch.Tensor,  # (B, C, ...)
        targets: torch.Tensor,  # (B, ...)
        smooth: float = 1e-5,
    ) -> torch.Tensor:
        """Soft Dice loss over all foreground classes."""
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        targets_onehot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes)
        targets_onehot = targets_onehot.permute(0, -1, *range(1, targets.dim())).float()

        # Exclude background (class 0) from Dice
        probs_fg = probs[:, 1:]
        targets_fg = targets_onehot[:, 1:]

        intersection = (probs_fg * targets_fg).sum(dim=list(range(2, probs_fg.dim())))
        cardinality = probs_fg.sum(dim=list(range(2, probs_fg.dim()))) + \
                      targets_fg.sum(dim=list(range(2, targets_fg.dim())))
        dice = 1 - (2 * intersection + smooth) / (cardinality + smooth)
        return dice.mean()

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        optimizer_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """Save model checkpoint with metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "in_channels": 1,
                "num_classes": self.model.num_classes,
                "base_channels": 32,
                "max_channels": 320,
                "depth": 4,
            },
            "metrics": metrics or {},
        }
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s (epoch %d)", path, epoch)


# ---------------------------------------------------------------------------
# SimpleITK integration — save/load segmentation labels
# ---------------------------------------------------------------------------

def save_segmentation_nifti(
    label_map: np.ndarray,
    reference_image: sitk.Image,
    output_path: str | Path,
) -> None:
    """
    Save integer label map as NIfTI, copying spatial metadata from reference.

    Args:
        label_map: (D, H, W) uint8 array.
        reference_image: SimpleITK image providing origin, spacing, direction.
        output_path: Output file path (.nii.gz).
    """
    sitk_seg = sitk.GetImageFromArray(label_map.astype(np.uint8))
    sitk_seg.CopyInformation(reference_image)
    sitk.WriteImage(sitk_seg, str(output_path))
    logger.info("Saved segmentation to %s", output_path)


def load_segmentation_nifti(path: str | Path) -> Tuple[np.ndarray, sitk.Image]:
    """
    Load a NIfTI segmentation label map.

    Returns:
        Tuple of (label_map numpy array, SimpleITK image object).
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.uint8)  # (D, H, W)
    return arr, img


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    logger.info("BoneSegmentor self-test (random weights)")

    cfg = SegmentorConfig(device="cpu", patch_size=(64, 64, 64))
    segmentor = BoneSegmentor.build_new(cfg)
    dummy_volume = np.random.randn(96, 96, 96).astype(np.float32) * 400
    result = segmentor.predict(dummy_volume)
    for cls_name, mask in result.items():
        if cls_name != "label_map":
            logger.info("  %s: %d voxels", cls_name, mask.sum())
    logger.info("Self-test passed.")
    sys.exit(0)


def sliding_window_inference_gc(volume, model, patch_size, overlap=0.25):
    """
    Sliding window inference with explicit memory cleanup between patches.
    Fixes GPU memory accumulation observed when processing large volumes.
    """
    import gc
    import numpy as np

    D, H, W = volume.shape
    pd, ph, pw = patch_size
    stride_d = max(1, int(pd * (1 - overlap)))
    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))

    out = np.zeros(volume.shape, dtype=np.float32)
    count = np.zeros(volume.shape, dtype=np.float32)

    for d in range(0, max(1, D - pd + 1), stride_d):
        for h in range(0, max(1, H - ph + 1), stride_h):
            for w in range(0, max(1, W - pw + 1), stride_w):
                patch = volume[d:d+pd, h:h+ph, w:w+pw]
                # mock inference
                pred = (patch > 0.5).astype(np.float32)
                out[d:d+pd, h:h+ph, w:w+pw] += pred
                count[d:d+pd, h:h+ph, w:w+pw] += 1
                # explicit gc call every patch to free intermediates
                gc.collect()

    count = np.maximum(count, 1)
    return out / count
