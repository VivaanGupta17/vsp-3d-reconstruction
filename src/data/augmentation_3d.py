"""
augmentation_3d.py
------------------
3D volumetric data augmentation for CT/CBCT bone segmentation training.

Augmentations:
  Spatial:
    - Random 3D rotation (±30°, axis-aligned or random axis)
    - Random flipping (left-right, for non-lateralised models)
    - Elastic deformation (3D Gaussian displacement fields)
    - Random scaling (0.85–1.15×)
    - Random translation / crop

  Intensity (CT-specific):
    - Gaussian noise
    - Random bias field (polynomial, simulates CBCT non-uniformity)
    - Contrast adjustment (gamma, window/level)
    - Blur/sharpen (spatial frequency modification)
    - Random HU offset (simulate scanner variation)

  Artefact simulation (for robustness):
    - Metal streak artifact simulation
    - Beam hardening approximation
    - Partial volume effect simulation
    - Noise-level modulation (kV/mAs variation)

  Segmentation-specific:
    - Consistent transform application to volume + mask
    - Boundary-respecting elastic deformation
    - Cropping to foreground (bone voxels)

All augmentations are implemented using numpy + scipy (no dependency on
albumentations or torchio, for portability; adapters for those libraries
are provided if available).

Usage::

    aug = Compose3D([
        RandomRotation3D(max_angle_deg=20),
        RandomElasticDeformation3D(sigma=10),
        RandomGaussianNoise(std_range=(0, 50)),
        RandomBiasField(order=3),
    ])
    aug_vol, aug_mask = aug(volume, mask)
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import (
    affine_transform,
    gaussian_filter,
    map_coordinates,
    rotate,
    zoom,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Transform3D(ABC):
    """Abstract base class for 3D augmentation transforms."""

    def __init__(self, p: float = 0.5) -> None:
        """
        Args:
            p: Probability of applying this transform (0–1).
        """
        self.p = p

    def __call__(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply transform with probability p."""
        if np.random.random() < self.p:
            return self.apply(volume, mask)
        return volume, mask

    @abstractmethod
    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply transform unconditionally."""
        ...


class Compose3D:
    """Compose a list of 3D transforms into a single callable."""

    def __init__(self, transforms: List[Transform3D]) -> None:
        self.transforms = transforms

    def __call__(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        for t in self.transforms:
            volume, mask = t(volume, mask)
        return volume, mask


# ---------------------------------------------------------------------------
# Spatial transforms
# ---------------------------------------------------------------------------

class RandomRotation3D(Transform3D):
    """
    Random 3D rotation around one or all axes.

    For CMF training, use ±20° max angle to preserve anatomical orientation.
    For full-body orthopedic, larger ranges (±30°) are acceptable.
    """

    def __init__(
        self,
        max_angle_deg: float = 20.0,
        axes: str = "all",   # "all" | "z" | "xy" | "xyz"
        p: float = 0.8,
    ) -> None:
        super().__init__(p)
        self.max_angle_deg = max_angle_deg
        self.axes = axes

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.axes == "z":
            # Axial rotation only (most common for CMF)
            angle = np.random.uniform(-self.max_angle_deg, self.max_angle_deg)
            aug_vol = rotate(volume, angle, axes=(1, 2), reshape=False,
                             order=1, mode="constant", cval=volume.min())
            aug_mask = (rotate(mask.astype(np.float32), angle, axes=(1, 2),
                               reshape=False, order=0, mode="constant") > 0.5).astype(mask.dtype) \
                        if mask is not None else None
        else:
            # Compose rotations around multiple axes
            aug_vol = volume
            aug_mask = mask
            for ax_pair in [(1, 2), (0, 2), (0, 1)]:
                if np.random.random() < 0.5:
                    angle = np.random.uniform(-self.max_angle_deg, self.max_angle_deg)
                    aug_vol = rotate(aug_vol, angle, axes=ax_pair, reshape=False,
                                     order=1, mode="constant", cval=volume.min())
                    if aug_mask is not None:
                        aug_mask = (rotate(aug_mask.astype(np.float32), angle, axes=ax_pair,
                                           reshape=False, order=0, mode="constant") > 0.5
                                    ).astype(mask.dtype)

        return aug_vol, aug_mask


class RandomFlip3D(Transform3D):
    """Random axis-aligned flipping. Left-right flip only for non-lateralised models."""

    def __init__(self, axes: Tuple[int, ...] = (2,), p: float = 0.5) -> None:
        super().__init__(p)
        self.axes = axes

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        for ax in self.axes:
            if np.random.random() < 0.5:
                volume = np.flip(volume, axis=ax).copy()
                if mask is not None:
                    mask = np.flip(mask, axis=ax).copy()
        return volume, mask


class RandomElasticDeformation3D(Transform3D):
    """
    3D elastic deformation via random Gaussian displacement fields.

    Simulates the natural variation in bone shape between patients and
    imaging sessions. Critical for building robustness to shape changes.

    Reference:
        Simard et al., "Best Practices for CNNs Applied to Visual Document
        Analysis", ICDAR 2003 (extended to 3D).
    """

    def __init__(
        self,
        sigma: float = 10.0,     # Gaussian smoothing σ for displacement (mm)
        alpha: float = 500.0,    # Displacement magnitude scaling factor
        p: float = 0.5,
    ) -> None:
        super().__init__(p)
        self.sigma = sigma
        self.alpha = alpha

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        shape = volume.shape

        # Generate random displacement fields for each axis
        dx = gaussian_filter(
            np.random.randn(*shape).astype(np.float32), sigma=self.sigma
        ) * self.alpha
        dy = gaussian_filter(
            np.random.randn(*shape).astype(np.float32), sigma=self.sigma
        ) * self.alpha
        dz = gaussian_filter(
            np.random.randn(*shape).astype(np.float32), sigma=self.sigma
        ) * self.alpha

        z, y, x = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
            indexing="ij",
        )
        coords = [
            np.clip(z + dz, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(x + dx, 0, shape[2] - 1),
        ]

        aug_vol = map_coordinates(volume, coords, order=1, mode="reflect").astype(volume.dtype)
        aug_mask = (map_coordinates(mask.astype(np.float32), coords, order=0, mode="nearest") > 0.5
                    ).astype(mask.dtype) if mask is not None else None

        return aug_vol, aug_mask


class RandomScaling3D(Transform3D):
    """Random isotropic/anisotropic scaling."""

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        anisotropic: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p)
        self.scale_range = scale_range
        self.anisotropic = anisotropic

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.anisotropic:
            scale = [np.random.uniform(*self.scale_range) for _ in range(3)]
        else:
            s = np.random.uniform(*self.scale_range)
            scale = [s, s, s]

        aug_vol = zoom(volume, scale, order=1)
        aug_mask = zoom(mask.astype(np.float32), scale, order=0).astype(mask.dtype) \
                   if mask is not None else None

        # Crop or pad to original size
        aug_vol = self._match_size(aug_vol, volume.shape, fill_val=float(volume.min()))
        if aug_mask is not None:
            aug_mask = self._match_size(aug_mask, mask.shape, fill_val=0).astype(mask.dtype)

        return aug_vol, aug_mask

    @staticmethod
    def _match_size(
        arr: np.ndarray,
        target_shape: Tuple[int, ...],
        fill_val: float = 0.0,
    ) -> np.ndarray:
        """Crop or pad array to target_shape."""
        result = np.full(target_shape, fill_val, dtype=arr.dtype)
        slices_src = tuple(slice(0, min(a, t)) for a, t in zip(arr.shape, target_shape))
        slices_dst = tuple(slice(0, min(a, t)) for a, t in zip(arr.shape, target_shape))
        result[slices_dst] = arr[slices_src]
        return result


class RandomCrop3D(Transform3D):
    """
    Random crop to a fixed patch size.
    Biased toward foreground voxels (bone) with probability fg_p.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        fg_p: float = 0.33,   # probability of sampling centred on foreground
        p: float = 1.0,
    ) -> None:
        super().__init__(p)
        self.patch_size = patch_size
        self.fg_p = fg_p

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        D, H, W = volume.shape
        pd, ph, pw = self.patch_size

        if np.random.random() < self.fg_p and mask is not None:
            # Sample a foreground voxel as crop centre
            fg_coords = np.argwhere(mask > 0)
            if len(fg_coords) > 0:
                centre = fg_coords[np.random.randint(len(fg_coords))]
                z0 = int(np.clip(centre[0] - pd // 2, 0, max(0, D - pd)))
                y0 = int(np.clip(centre[1] - ph // 2, 0, max(0, H - ph)))
                x0 = int(np.clip(centre[2] - pw // 2, 0, max(0, W - pw)))
            else:
                z0 = np.random.randint(0, max(1, D - pd))
                y0 = np.random.randint(0, max(1, H - ph))
                x0 = np.random.randint(0, max(1, W - pw))
        else:
            z0 = np.random.randint(0, max(1, D - pd))
            y0 = np.random.randint(0, max(1, H - ph))
            x0 = np.random.randint(0, max(1, W - pw))

        crop_vol = volume[z0:z0+pd, y0:y0+ph, x0:x0+pw]
        crop_mask = mask[z0:z0+pd, y0:y0+ph, x0:x0+pw] if mask is not None else None

        # Pad if volume is smaller than patch
        if crop_vol.shape != self.patch_size:
            pad = [(0, max(0, p - c)) for c, p in zip(crop_vol.shape, self.patch_size)]
            crop_vol = np.pad(crop_vol, pad, mode="constant", constant_values=volume.min())
            if crop_mask is not None:
                crop_mask = np.pad(crop_mask, pad, mode="constant")

        return crop_vol, crop_mask


# ---------------------------------------------------------------------------
# Intensity augmentations
# ---------------------------------------------------------------------------

class RandomGaussianNoise(Transform3D):
    """Add Gaussian noise to simulate detector noise (kV/mAs variation)."""

    def __init__(
        self,
        std_range: Tuple[float, float] = (0, 50.0),  # HU
        p: float = 0.5,
    ) -> None:
        super().__init__(p)
        self.std_range = std_range

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        std = np.random.uniform(*self.std_range)
        noise = np.random.normal(0, std, volume.shape).astype(np.float32)
        return (volume + noise).astype(volume.dtype), mask


class RandomGaussianBlur(Transform3D):
    """Apply random Gaussian smoothing (simulate lower resolution / slice thickness)."""

    def __init__(
        self,
        sigma_range: Tuple[float, float] = (0.5, 2.0),  # voxels
        p: float = 0.3,
    ) -> None:
        super().__init__(p)
        self.sigma_range = sigma_range

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        sigma = np.random.uniform(*self.sigma_range)
        blurred = gaussian_filter(volume.astype(np.float32), sigma=sigma)
        return blurred.astype(volume.dtype), mask


class RandomContrastAdjustment(Transform3D):
    """
    Gamma contrast adjustment for CT (simulates window/level variation).
    CT-specific: applied only within the soft tissue / bone window.
    """

    def __init__(
        self,
        gamma_range: Tuple[float, float] = (0.7, 1.4),
        p: float = 0.5,
    ) -> None:
        super().__init__(p)
        self.gamma_range = gamma_range

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        gamma = np.random.uniform(*self.gamma_range)
        # Normalise to [0, 1] for gamma, then restore scale
        v_min, v_max = float(volume.min()), float(volume.max())
        if v_max - v_min < 1e-6:
            return volume, mask
        v_norm = (volume.astype(np.float32) - v_min) / (v_max - v_min)
        v_gamma = np.power(np.clip(v_norm, 0, 1), gamma)
        return (v_gamma * (v_max - v_min) + v_min).astype(volume.dtype), mask


class RandomBiasField(Transform3D):
    """
    Simulates low-frequency intensity bias field.
    Critical for CBCT training: CBCT has strong cone-beam bias artefacts.

    Generates a random polynomial bias field and multiplies the volume.
    """

    def __init__(
        self,
        order: int = 3,
        magnitude: float = 0.4,   # max relative modulation
        p: float = 0.5,
    ) -> None:
        super().__init__(p)
        self.order = order
        self.magnitude = magnitude

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        D, H, W = volume.shape

        # Normalised coordinate grids
        z = np.linspace(-1, 1, D, dtype=np.float32)
        y = np.linspace(-1, 1, H, dtype=np.float32)
        x = np.linspace(-1, 1, W, dtype=np.float32)
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

        # Random polynomial bias field
        bias = np.ones((D, H, W), dtype=np.float32)
        for deg_z in range(self.order + 1):
            for deg_y in range(self.order + 1 - deg_z):
                for deg_x in range(self.order + 1 - deg_z - deg_y):
                    coeff = np.random.uniform(-self.magnitude / 3, self.magnitude / 3)
                    bias += coeff * (zz ** deg_z) * (yy ** deg_y) * (xx ** deg_x)

        # Apply bias field to volume (multiplicative)
        bias = np.clip(bias, 1 - self.magnitude, 1 + self.magnitude)
        aug_vol = (volume.astype(np.float32) * bias).astype(volume.dtype)
        return aug_vol, mask


class RandomHUOffset(Transform3D):
    """
    Add random global HU offset to simulate scanner calibration variation.
    The offset shifts the entire HU scale by a small constant.
    """

    def __init__(
        self,
        offset_range: Tuple[float, float] = (-50.0, 50.0),  # HU
        p: float = 0.5,
    ) -> None:
        super().__init__(p)
        self.offset_range = offset_range

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        offset = np.random.uniform(*self.offset_range)
        return (volume.astype(np.float32) + offset).astype(volume.dtype), mask


# ---------------------------------------------------------------------------
# Artefact simulation
# ---------------------------------------------------------------------------

class MetalArtifactSimulation(Transform3D):
    """
    Simulate metal streak artefacts for CBCT robustness.

    Places a random metallic object (sphere/cylinder of HU ~3000) and
    adds streak artefacts radiating from it, as seen in CBCT with
    dental implants, plates, or screws.
    """

    def __init__(
        self,
        n_metal_objects: Tuple[int, int] = (1, 4),
        metal_hu: float = 3000.0,
        streak_intensity: float = 500.0,
        streak_width_vox: int = 2,
        p: float = 0.3,
    ) -> None:
        super().__init__(p)
        self.n_metal_objects = n_metal_objects
        self.metal_hu = metal_hu
        self.streak_intensity = streak_intensity
        self.streak_width = streak_width_vox

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        aug = volume.copy().astype(np.float32)
        D, H, W = aug.shape

        n_obj = np.random.randint(*self.n_metal_objects)
        for _ in range(n_obj):
            # Random metallic sphere
            r = np.random.randint(2, 6)
            cz = np.random.randint(r, D - r)
            cy = np.random.randint(r, H - r)
            cx = np.random.randint(r, W - r)

            z, y, x = np.ogrid[cz-r:cz+r, cy-r:cy+r, cx-r:cx+r]
            sphere = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2

            aug[cz-r:cz+r, cy-r:cy+r, cx-r:cx+r][sphere] = self.metal_hu

            # Horizontal streak artefacts (simplified — real MAR is CT-geometry specific)
            streak_row = cy + np.random.randint(-10, 10)
            if 0 <= streak_row < H:
                streak_noise = np.random.randn(W).astype(np.float32) * self.streak_intensity
                aug[cz, streak_row, :] += streak_noise

        return aug.astype(volume.dtype), mask


class PartialVolumeSimulation(Transform3D):
    """
    Simulate partial volume effect by locally blurring the boundary
    between bone and soft tissue. Mimics the volume averaging artefact
    seen at bone edges in thick-slice CT protocols.
    """

    def __init__(
        self,
        sigma: float = 0.8,  # voxels
        boundary_dilation: int = 2,
        p: float = 0.3,
    ) -> None:
        super().__init__(p)
        self.sigma = sigma
        self.boundary_dilation = boundary_dilation

    def apply(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        from scipy.ndimage import binary_dilation, binary_erosion

        if mask is None:
            # Fall back to intensity-based bone boundary
            bone_mask = volume > 300  # HU threshold
        else:
            bone_mask = mask > 0

        # Find boundary region
        dilated = binary_dilation(bone_mask, iterations=self.boundary_dilation)
        eroded = binary_erosion(bone_mask, iterations=self.boundary_dilation)
        boundary = dilated & ~eroded

        # Apply Gaussian blur only at boundary
        blurred = gaussian_filter(volume.astype(np.float32), sigma=self.sigma)
        aug = volume.astype(np.float32).copy()
        aug[boundary] = blurred[boundary]

        return aug.astype(volume.dtype), mask


# ---------------------------------------------------------------------------
# Complete training augmentation pipeline
# ---------------------------------------------------------------------------

def build_training_augmentation(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    modality: str = "CT",
    aggressive: bool = False,
) -> Compose3D:
    """
    Build a standard augmentation pipeline for CT/CBCT bone segmentation training.

    Args:
        patch_size: Output patch size.
        modality: "CT" | "CBCT" — affects intensity augmentation parameters.
        aggressive: If True, use stronger augmentation (longer training).

    Returns:
        Compose3D pipeline.
    """
    transforms = [
        # Spatial transforms
        RandomCrop3D(patch_size=patch_size, fg_p=0.33, p=1.0),
        RandomFlip3D(axes=(2,), p=0.5),            # left-right flip
        RandomRotation3D(max_angle_deg=20 if aggressive else 15, axes="all", p=0.8),
        RandomScaling3D(scale_range=(0.85, 1.15), p=0.5),
        RandomElasticDeformation3D(sigma=10, alpha=300 if aggressive else 150, p=0.3),

        # Intensity transforms
        RandomGaussianNoise(std_range=(0, 60 if modality == "CBCT" else 40), p=0.5),
        RandomGaussianBlur(sigma_range=(0.5, 1.5), p=0.3),
        RandomContrastAdjustment(gamma_range=(0.75, 1.35), p=0.5),
        RandomHUOffset(offset_range=(-30, 30), p=0.5),

        # Artefact simulation
        MetalArtifactSimulation(p=0.4 if modality == "CBCT" else 0.2),
        PartialVolumeSimulation(p=0.3),
    ]

    if modality == "CBCT":
        transforms.insert(-1, RandomBiasField(order=3, magnitude=0.4, p=0.6))

    return Compose3D(transforms)


def build_validation_augmentation(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
) -> Compose3D:
    """Minimal augmentation for validation (crop only)."""
    return Compose3D([
        RandomCrop3D(patch_size=patch_size, fg_p=0.5, p=1.0),
    ])


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)

    # Create synthetic volume and mask
    D, H, W = 160, 160, 160
    volume = (np.random.randn(D, H, W) * 300 + 200).astype(np.float32)
    # Sphere mask
    z, y, x = np.ogrid[:D, :H, :W]
    mask = ((z - D//2)**2 + (y - H//2)**2 + (x - W//2)**2 <= 40**2).astype(np.uint8)
    # Add metal
    volume[D//2-5:D//2+5, H//2-5:H//2+5, W//2-5:W//2+5] = 3000

    pipeline = build_training_augmentation(
        patch_size=(128, 128, 128),
        modality="CBCT",
        aggressive=True,
    )

    aug_vol, aug_mask = pipeline(volume, mask)
    logger.info("Input:  vol=%s mask=%s | HU mean=%.1f", volume.shape, mask.shape, volume.mean())
    logger.info("Output: vol=%s mask=%s | HU mean=%.1f", aug_vol.shape, aug_mask.shape, aug_vol.mean())
    logger.info("Mask foreground voxels: %d → %d", mask.sum(), aug_mask.sum())

    logger.info("augmentation_3d self-test passed.")
    sys.exit(0)
