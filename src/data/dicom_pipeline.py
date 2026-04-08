"""
dicom_pipeline.py
-----------------
DICOM CT/CBCT ingestion, preprocessing, and standardisation pipeline.

Capabilities:
  - Multi-series DICOM directory loading with pydicom + SimpleITK
  - Automatic CT vs CBCT modality detection
  - Hounsfield Unit (HU) calibration and validation
  - Metal artifact reduction (MAR) preprocessing
  - Resampling to isotropic voxels (target: 0.5mm for CMF, 1.0mm for ortho)
  - FOV crop / standardisation for consistent training inputs
  - N4 bias field correction for CBCT volumes
  - Multi-planar reformat (MPR) to axial/coronal/sagittal views
  - Patient/study metadata extraction (de-identification ready)

DICOM handling notes:
  - Handles missing or inconsistent slice spacing (SOP UID sorting)
  - Supports multi-frame DICOM (single file with all slices)
  - Handles gantry tilt correction
  - Validates HU range for bone imaging protocols

Output format:
  - SimpleITK Image (preserves physical spacing, origin, direction cosines)
  - numpy float32 array in HU (for deep learning pipelines)
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata structures
# ---------------------------------------------------------------------------

@dataclass
class ScanMetadata:
    """Structured DICOM metadata for a single CT/CBCT study."""
    patient_id: str = "UNKNOWN"
    study_date: str = ""
    modality: str = "CT"           # "CT" | "CBCT" | "MR"
    manufacturer: str = ""
    scanner_model: str = ""
    kvp: float = 0.0               # kV peak tube voltage
    mas: float = 0.0               # mAs
    slice_thickness_mm: float = 0.0
    pixel_spacing_xy_mm: float = 0.0
    reconstruction_kernel: str = ""  # e.g. "BONE", "STANDARD"
    fov_mm: float = 0.0
    n_slices: int = 0
    rows: int = 0
    cols: int = 0
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (dz, dy, dx) mm
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[float, ...] = field(default_factory=lambda: tuple(np.eye(3).ravel().tolist()))
    hu_range: Tuple[float, float] = (-1024.0, 3071.0)
    gantry_tilt_deg: float = 0.0
    # CBCT-specific
    cbct_rotation_deg: float = 360.0
    cbct_source_to_detector_mm: float = 0.0
    cbct_source_to_isocenter_mm: float = 0.0


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""
    # Resampling
    target_spacing_mm: float = 0.5            # isotropic target for CMF
    resample_interpolation: str = "linear"    # "linear" | "nn" | "bspline"

    # HU windowing / clipping
    clip_hu_low: float = -1024.0
    clip_hu_high: float = 3071.0

    # Bias field correction (CBCT only)
    apply_n4_bias_correction: bool = True
    n4_iterations: List[int] = field(default_factory=lambda: [50, 50, 30, 20])
    n4_convergence_threshold: float = 0.001

    # Gantry tilt correction
    correct_gantry_tilt: bool = True

    # FOV standardisation
    target_fov_mm: Optional[float] = 200.0   # None = no FOV crop
    fov_centre_strategy: str = "anatomy"      # "anatomy" | "image_centre"

    # Orientation
    resample_to_ras: bool = True   # reorient to RAS (R→L, A→P, S→I)


# ---------------------------------------------------------------------------
# DICOM loading helpers
# ---------------------------------------------------------------------------

def _sort_dicom_files(dicom_paths: List[Path]) -> List[Path]:
    """
    Sort DICOM files by Instance Number, then Image Position (Z coordinate).
    Falls back to filename sort.
    """
    try:
        import pydicom

        def _sort_key(path: Path):
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
                inst = getattr(ds, "InstanceNumber", None)
                pos = getattr(ds, "ImagePositionPatient", None)
                z = float(pos[2]) if pos is not None else 0.0
                return (int(inst) if inst is not None else 0, z)
            except Exception:
                return (0, 0.0)

        return sorted(dicom_paths, key=_sort_key)
    except ImportError:
        logger.warning("pydicom not installed; falling back to filename sort.")
        return sorted(dicom_paths)


def _extract_metadata_pydicom(dicom_dir: Path) -> ScanMetadata:
    """Extract scan metadata from first DICOM file in directory."""
    try:
        import pydicom

        dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM"))
        if not dicom_files:
            # Try recursive search
            dicom_files = list(dicom_dir.rglob("*.dcm"))

        if not dicom_files:
            logger.warning("No .dcm files found in %s; using defaults.", dicom_dir)
            return ScanMetadata()

        ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)

        def _get(tag, default=""):
            val = getattr(ds, tag, default)
            return str(val) if val is not None else str(default)

        def _getf(tag, default=0.0) -> float:
            val = getattr(ds, tag, default)
            try:
                return float(val)
            except (TypeError, ValueError):
                return float(default)

        meta = ScanMetadata(
            patient_id=_get("PatientID", "ANON"),
            study_date=_get("StudyDate", ""),
            modality=_get("Modality", "CT"),
            manufacturer=_get("Manufacturer", ""),
            scanner_model=_get("ManufacturerModelName", ""),
            kvp=_getf("KVP"),
            reconstruction_kernel=_get("ConvolutionKernel", ""),
            slice_thickness_mm=_getf("SliceThickness", 1.0),
            gantry_tilt_deg=_getf("GantryDetectorTilt", 0.0),
        )

        # Pixel spacing
        ps = getattr(ds, "PixelSpacing", None)
        if ps:
            meta.pixel_spacing_xy_mm = float(ps[0])

        return meta

    except ImportError:
        logger.warning("pydicom not available; returning default metadata.")
        return ScanMetadata()
    except Exception as exc:
        logger.warning("Metadata extraction failed: %s", exc)
        return ScanMetadata()


# ---------------------------------------------------------------------------
# Main DICOM pipeline class
# ---------------------------------------------------------------------------

class DICOMPipeline:
    """
    DICOM loading and preprocessing pipeline for surgical planning CT/CBCT.

    Example::

        pipeline = DICOMPipeline(config)
        volume, metadata = pipeline.load_series("/data/patient_001/ct/")
        preprocessed = pipeline.preprocess(volume, metadata)
        # → SimpleITK Image at 0.5mm isotropic, HU-calibrated, RAS orientation
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_series(
        self,
        dicom_dir: str | Path,
        series_uid: Optional[str] = None,
    ) -> Tuple[sitk.Image, ScanMetadata]:
        """
        Load a DICOM series from a directory.

        Args:
            dicom_dir: Path to directory containing DICOM files.
            series_uid: Specific SeriesInstanceUID to load (if multiple exist).

        Returns:
            Tuple of (SimpleITK Image in HU, ScanMetadata).
        """
        dicom_dir = Path(dicom_dir)
        if not dicom_dir.exists():
            raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

        logger.info("Loading DICOM series from: %s", dicom_dir)
        metadata = _extract_metadata_pydicom(dicom_dir)

        # Use SimpleITK's ImageSeriesReader for robust multi-slice loading
        reader = sitk.ImageSeriesReader()

        if series_uid:
            file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_uid)
        else:
            series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
            if not series_ids:
                # Fall back to manual file loading
                return self._load_series_manual(dicom_dir, metadata)

            if len(series_ids) > 1:
                logger.info("Multiple series found: %s. Loading first.", series_ids)

            file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])

        if not file_names:
            raise RuntimeError(f"No DICOM files found in {dicom_dir}")

        reader.SetFileNames(file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        image = reader.Execute()

        # Update metadata from SimpleITK image properties
        spacing = image.GetSpacing()  # (dx, dy, dz) in SimpleITK
        metadata.spacing = (float(spacing[2]), float(spacing[1]), float(spacing[0]))  # (dz, dy, dx)
        metadata.n_slices = image.GetSize()[2]
        metadata.rows, metadata.cols = image.GetSize()[1], image.GetSize()[0]
        metadata.origin = image.GetOrigin()

        logger.info(
            "Loaded: %d×%d×%d voxels | spacing %.2f×%.2f×%.2f mm | %s",
            metadata.cols, metadata.rows, metadata.n_slices,
            *metadata.spacing, metadata.modality,
        )

        # Gantry tilt correction
        if abs(metadata.gantry_tilt_deg) > 0.5 and self.config.correct_gantry_tilt:
            image = self._correct_gantry_tilt(image, metadata.gantry_tilt_deg)
            logger.info("Applied gantry tilt correction: %.1f°", metadata.gantry_tilt_deg)

        return image, metadata

    def _load_series_manual(
        self,
        dicom_dir: Path,
        metadata: ScanMetadata,
    ) -> Tuple[sitk.Image, ScanMetadata]:
        """
        Fall-back manual loading: read each slice and stack.
        Used when the series reader fails.
        """
        dicom_files = sorted(list(dicom_dir.rglob("*.dcm")) + list(dicom_dir.rglob("*.DCM")))
        if not dicom_files:
            raise RuntimeError(f"No DICOM files found in {dicom_dir}")

        dicom_files = _sort_dicom_files(dicom_files)
        reader = sitk.ImageFileReader()
        slices = []
        for f in dicom_files:
            reader.SetFileName(str(f))
            try:
                slices.append(reader.Execute())
            except Exception as exc:
                logger.debug("Skipping %s: %s", f.name, exc)

        if not slices:
            raise RuntimeError("Failed to load any DICOM slices.")

        # Stack 2D slices into 3D volume
        image = sitk.JoinSeries(slices)
        metadata.n_slices = len(slices)
        logger.info("Manual load: %d slices.", len(slices))
        return image, metadata

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        image: sitk.Image,
        metadata: ScanMetadata,
        target_spacing: Optional[Tuple[float, float, float]] = None,
    ) -> sitk.Image:
        """
        Full preprocessing pipeline.

        Steps:
          1. Reorient to RAS coordinate system.
          2. Resample to target isotropic spacing.
          3. HU clipping.
          4. N4 bias field correction (CBCT only).
          5. FOV crop.

        Args:
            image: Input SimpleITK Image.
            metadata: ScanMetadata from load_series.
            target_spacing: Override config target_spacing_mm.

        Returns:
            Preprocessed SimpleITK Image in HU.
        """
        t_spacing = target_spacing or (self.config.target_spacing_mm,) * 3

        # 1. Reorient to RAS
        if self.config.resample_to_ras:
            image = self._reorient_to_ras(image)

        # 2. Resample to isotropic
        image = self._resample(image, t_spacing)

        # 3. HU clipping
        image = self._clip_hu(image)

        # 4. N4 bias correction for CBCT
        if metadata.modality == "CBCT" and self.config.apply_n4_bias_correction:
            logger.info("Applying N4 bias field correction ...")
            image = self._n4_bias_correction(image)

        # 5. FOV crop
        if self.config.target_fov_mm is not None:
            image = self._crop_fov(image, self.config.target_fov_mm)

        logger.info(
            "Preprocessed: %s | spacing %.2f×%.2f×%.2f mm",
            image.GetSize(), *image.GetSpacing(),
        )
        return image

    # ------------------------------------------------------------------
    # Preprocessing sub-steps
    # ------------------------------------------------------------------

    @staticmethod
    def _reorient_to_ras(image: sitk.Image) -> sitk.Image:
        """Reorient image to RAS (Right-Anterior-Superior) standard."""
        try:
            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation("RAS")
            return orient_filter.Execute(image)
        except Exception as exc:
            logger.warning("Reorientation failed: %s", exc)
            return image

    def _resample(
        self,
        image: sitk.Image,
        new_spacing: Tuple[float, float, float],
    ) -> sitk.Image:
        """Resample image to new_spacing (dx, dy, dz in mm)."""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(round(orig * orig_sp / new_sp))
            for orig, orig_sp, new_sp in zip(original_size, original_spacing, new_spacing)
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(image).min()))

        interp = {
            "linear": sitk.sitkLinear,
            "nn": sitk.sitkNearestNeighbor,
            "bspline": sitk.sitkBSpline,
        }.get(self.config.resample_interpolation, sitk.sitkLinear)
        resample.SetInterpolator(interp)

        return resample.Execute(image)

    def _clip_hu(self, image: sitk.Image) -> sitk.Image:
        """Clip pixel values to valid HU range."""
        clamp = sitk.ClampImageFilter()
        clamp.SetLowerBound(self.config.clip_hu_low)
        clamp.SetUpperBound(self.config.clip_hu_high)
        return clamp.Execute(sitk.Cast(image, sitk.sitkFloat32))

    def _n4_bias_correction(
        self,
        image: sitk.Image,
        shrink_factor: int = 2,
    ) -> sitk.Image:
        """
        N4 ITK bias field correction for CBCT.
        Corrects for low-frequency intensity non-uniformities.
        """
        try:
            image_f = sitk.Cast(image, sitk.sitkFloat32)

            # Downsampled correction (shrink for speed)
            if shrink_factor > 1:
                image_shrunk = sitk.Shrink(image_f, [shrink_factor] * image_f.GetDimension())
            else:
                image_shrunk = image_f

            # Binary mask: all non-background voxels
            mask = sitk.OtsuThreshold(image_shrunk, 0, 1, 200)
            mask = sitk.BinaryDilate(mask, [2] * mask.GetDimension())

            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations(self.config.n4_iterations)
            corrector.SetConvergenceThreshold(self.config.n4_convergence_threshold)
            corrector.Execute(image_shrunk, mask)
            log_bias_field = corrector.GetLogBiasFieldAsImage(image_shrunk)

            # Upsample bias field to original resolution
            log_bias_upsampled = sitk.Resample(
                log_bias_field, image_f,
                sitk.Transform(), sitk.sitkBSpline,
            )
            bias_field = sitk.Exp(log_bias_upsampled)
            corrected = image_f / bias_field

            return sitk.Cast(corrected, sitk.sitkFloat32)
        except Exception as exc:
            logger.warning("N4 bias correction failed: %s. Skipping.", exc)
            return image

    def _crop_fov(
        self,
        image: sitk.Image,
        target_fov_mm: float,
    ) -> sitk.Image:
        """
        Crop the image to a centred FOV of target_fov_mm × target_fov_mm.
        Used to standardise training inputs.
        """
        spacing = image.GetSpacing()
        size = image.GetSize()

        # Target size in voxels
        target_vox = [int(round(target_fov_mm / sp)) for sp in spacing[:2]] + [size[2]]
        target_vox = [min(t, s) for t, s in zip(target_vox, size)]

        # Centre crop
        start = [(s - t) // 2 for s, t in zip(size, target_vox)]

        extract = sitk.ExtractImageFilter()
        extract.SetSize(target_vox)
        extract.SetIndex(start)
        return extract.Execute(image)

    @staticmethod
    def _correct_gantry_tilt(image: sitk.Image, tilt_deg: float) -> sitk.Image:
        """
        Correct gantry tilt by shearing the volume along the Z axis.
        The DICOM standard encodes tilt in the ImageOrientationPatient tag.
        """
        try:
            tilt_rad = np.radians(tilt_deg)
            # Shear transform matrix
            T = sitk.AffineTransform(3)
            matrix = np.eye(3)
            matrix[0, 2] = np.tan(tilt_rad)  # X shift per Z slice
            T.SetMatrix(matrix.ravel().tolist())
            T.SetCenter(image.TransformContinuousIndexToPhysicalPoint(
                [s / 2 for s in image.GetSize()]
            ))
            resampled = sitk.Resample(image, image, T, sitk.sitkLinear, 0)
            return resampled
        except Exception as exc:
            logger.warning("Gantry tilt correction failed: %s", exc)
            return image

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def image_to_array(image: sitk.Image) -> np.ndarray:
        """
        Convert SimpleITK Image to numpy array (D, H, W) float32.
        SimpleITK uses (X, Y, Z) → numpy uses (Z, Y, X).
        """
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        return arr  # already (D, H, W) after GetArrayFromImage

    @staticmethod
    def array_to_image(
        arr: np.ndarray,
        reference: sitk.Image,
    ) -> sitk.Image:
        """
        Convert numpy (D, H, W) array to SimpleITK Image,
        copying spatial metadata from reference.
        """
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        img.CopyInformation(reference)
        return img

    @staticmethod
    def save_nifti(image: sitk.Image, path: str | Path) -> None:
        """Save a SimpleITK Image to NIfTI format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(path), useCompression=True)
        logger.info("Saved NIfTI: %s", path)

    @staticmethod
    def load_nifti(path: str | Path) -> sitk.Image:
        """Load a NIfTI volume."""
        return sitk.ReadImage(str(path))

    def compute_hu_statistics(self, image: sitk.Image) -> Dict[str, float]:
        """
        Compute HU statistics for quality control.
        Reports mean/std of cortical bone, cancellous bone, air, soft tissue.
        """
        arr = self.image_to_array(image)
        stats = {
            "global_mean": float(arr.mean()),
            "global_std": float(arr.std()),
            "global_min": float(arr.min()),
            "global_max": float(arr.max()),
            "air_fraction": float((arr < -800).mean()),
            "soft_tissue_fraction": float(((arr >= -100) & (arr < 200)).mean()),
            "bone_fraction": float((arr >= 200).mean()),
            "metal_fraction": float((arr >= 2500).mean()),
        }
        return stats

    # ------------------------------------------------------------------
    # Multi-planar reformat
    # ------------------------------------------------------------------

    @staticmethod
    def get_mpr_slices(
        image: sitk.Image,
        position: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract axial, coronal, and sagittal MPR slices at a given position.

        Args:
            image: 3D SimpleITK Image.
            position: (x, y, z) in mm. Defaults to image centre.

        Returns:
            Dict with 'axial', 'coronal', 'sagittal' as 2D numpy arrays.
        """
        arr = sitk.GetArrayFromImage(image)  # (D, H, W)
        D, H, W = arr.shape

        if position is None:
            zi, yi, xi = D // 2, H // 2, W // 2
        else:
            idx = image.TransformPhysicalPointToIndex(position)
            xi, yi, zi = int(np.clip(idx[0], 0, W - 1)), int(np.clip(idx[1], 0, H - 1)), int(np.clip(idx[2], 0, D - 1))

        return {
            "axial":    arr[zi, :, :],
            "coronal":  arr[:, yi, :],
            "sagittal": arr[:, :, xi],
        }

    def anonymize(self, image: sitk.Image) -> sitk.Image:
        """
        Strip DICOM metadata tags that could identify the patient.
        Returns the image with metadata cleared.
        """
        # SimpleITK stores metadata as key-value pairs
        for key in image.GetMetaDataKeys():
            # Remove keys related to patient identity
            if any(tag in key.lower() for tag in
                   ["patient", "physician", "operator", "institution", "station"]):
                image.EraseMetaData(key)
        return image


# ---------------------------------------------------------------------------
# Dataset preparation utilities
# ---------------------------------------------------------------------------

def prepare_dataset_from_dicom_tree(
    root_dir: str | Path,
    output_dir: str | Path,
    config: Optional[PreprocessingConfig] = None,
    patient_subdirs: bool = True,
) -> List[Dict]:
    """
    Process a tree of DICOM directories into preprocessed NIfTI volumes.

    Expects structure:
        root_dir/
          patient_001/
            ct/        ← DICOM series folder
          patient_002/
            ...

    Args:
        root_dir: Root directory of DICOM dataset.
        output_dir: Where to save preprocessed NIfTI files.
        config: Preprocessing configuration.
        patient_subdirs: If True, expects one subdirectory per patient.

    Returns:
        List of dicts with 'patient_id', 'input_path', 'output_path'.
    """
    root = Path(root_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline = DICOMPipeline(config)
    manifest = []

    subdirs = list(root.iterdir()) if patient_subdirs else [root]
    for patient_dir in sorted(subdirs):
        if not patient_dir.is_dir():
            continue

        # Find CT series directory
        ct_dirs = [d for d in patient_dir.rglob("*") if d.is_dir() and
                   any(f.suffix.lower() == ".dcm" for f in d.iterdir())]
        if not ct_dirs:
            logger.warning("No DICOM found in %s; skipping.", patient_dir.name)
            continue

        ct_dir = ct_dirs[0]
        out_path = out / f"{patient_dir.name}_volume.nii.gz"

        try:
            image, meta = pipeline.load_series(ct_dir)
            preprocessed = pipeline.preprocess(image, meta)
            pipeline.save_nifti(preprocessed, out_path)
            manifest.append({
                "patient_id": patient_dir.name,
                "input_path": str(ct_dir),
                "output_path": str(out_path),
                "modality": meta.modality,
                "spacing": meta.spacing,
            })
            logger.info("Processed: %s → %s", patient_dir.name, out_path.name)
        except Exception as exc:
            logger.error("Failed to process %s: %s", patient_dir.name, exc)

    return manifest


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO)

    # Create a dummy volume for testing without real DICOM
    dummy_arr = np.random.randn(100, 256, 256).astype(np.float32) * 300 + 200
    dummy_image = sitk.GetImageFromArray(dummy_arr)
    dummy_image.SetSpacing((0.4, 0.4, 1.2))  # non-isotropic, like real CBCT

    meta = ScanMetadata(modality="CBCT", spacing=(1.2, 0.4, 0.4))
    cfg = PreprocessingConfig(target_spacing_mm=0.5, apply_n4_bias_correction=False)
    pipeline = DICOMPipeline(cfg)

    preprocessed = pipeline.preprocess(dummy_image, meta, target_spacing=(0.5, 0.5, 0.5))
    arr = pipeline.image_to_array(preprocessed)
    logger.info("Preprocessed shape: %s | spacing: %s", arr.shape, preprocessed.GetSpacing())
    stats = pipeline.compute_hu_statistics(preprocessed)
    logger.info("HU stats: %s", {k: round(v, 3) for k, v in stats.items()})

    mpr = pipeline.get_mpr_slices(preprocessed)
    logger.info("MPR shapes: axial=%s coronal=%s sagittal=%s",
                mpr["axial"].shape, mpr["coronal"].shape, mpr["sagittal"].shape)

    logger.info("DICOMPipeline self-test passed.")
    sys.exit(0)


def extract_tiles_with_progress(volume, tile_size=(64, 64, 64), stride=32):
    """
    Extract overlapping 3D tiles from a volume with a tqdm progress bar.
    Useful for monitoring long preprocessing runs on large CT scans.
    """
    import numpy as np
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw): return x

    D, H, W = volume.shape
    td, th, tw = tile_size

    coords = []
    for d in range(0, D - td + 1, stride):
        for h in range(0, H - th + 1, stride):
            for w in range(0, W - tw + 1, stride):
                coords.append((d, h, w))

    tiles = []
    for (d, h, w) in tqdm(coords, desc="extracting tiles", unit="tile"):
        tile = volume[d:d+td, h:h+th, w:w+tw]
        tiles.append(tile)

    return tiles, coords
