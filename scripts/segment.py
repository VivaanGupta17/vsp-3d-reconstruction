#!/usr/bin/env python3
"""
segment.py — CLI for 3D bone segmentation

Usage:
    python scripts/segment.py --input /data/patient_001/ct/ --output /outputs/patient_001/
    python scripts/segment.py --input volume.nii.gz --output seg/ --model models/bone_v2.pt --modality CBCT
    python scripts/segment.py --input /data/ --output /outputs/ --batch  # batch mode

Arguments:
    --input PATH      DICOM directory, NIfTI file, or root dir (batch mode)
    --output DIR      Output directory (will be created if absent)
    --model PATH      Path to bone_segmentor checkpoint [default: models/bone_segmentor_v2.pt]
    --cmf-model PATH  Path to CMF/mandible segmentor checkpoint (uses CMF-specific model if provided)
    --modality STR    CT or CBCT [auto-detected if not specified]
    --spacing FLOAT   Target isotropic spacing in mm [default: 0.5]
    --device STR      cuda | cpu [default: cuda if available]
    --batch           Batch mode: process all subdirectories of --input
    --config PATH     YAML config file [default: configs/cmf_planning_config.yaml]
    --no-postproc     Skip post-processing (component analysis, morphological ops)
    --save-probabilities  Also save probability maps as NIfTI
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dicom_pipeline import DICOMPipeline, PreprocessingConfig
from src.segmentation.bone_segmentor import BoneSegmentor, SegmentorConfig, save_segmentation_nifti
from src.segmentation.mandible_segmentor import MandibleSegmentor, CMFSegmentorConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("segment")


def parse_args():
    parser = argparse.ArgumentParser(
        description="VSP-3D: 3D bone segmentation from CT/CBCT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  required=True,  help="DICOM directory or NIfTI file (or root dir in batch mode)")
    parser.add_argument("--output", required=True,  help="Output directory")
    parser.add_argument("--model",  default="models/bone_segmentor_v2.pt")
    parser.add_argument("--cmf-model", default=None, help="CMF coarse model checkpoint")
    parser.add_argument("--cmf-fine-model", default=None, help="CMF fine model checkpoint")
    parser.add_argument("--modality", choices=["CT", "CBCT"], default=None)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--config", default="configs/cmf_planning_config.yaml")
    parser.add_argument("--no-postproc", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--patch-size", nargs=3, type=int, default=[128, 128, 128])
    return parser.parse_args()


def load_volume(input_path: Path, pipeline: DICOMPipeline, spacing: float, modality: str = None):
    """Load a DICOM series or NIfTI file and preprocess."""
    if input_path.suffix in (".gz", ".nii"):
        logger.info("Loading NIfTI: %s", input_path)
        image = sitk.ReadImage(str(input_path))
        metadata_mod = modality or "CT"
        from src.data.dicom_pipeline import ScanMetadata
        metadata = ScanMetadata(modality=metadata_mod)
    else:
        logger.info("Loading DICOM: %s", input_path)
        image, metadata = pipeline.load_series(input_path)

    if modality:
        metadata.modality = modality

    preprocessed = pipeline.preprocess(image, metadata, target_spacing=(spacing,) * 3)
    volume_np = pipeline.image_to_array(preprocessed)
    return volume_np, preprocessed, metadata


def run_segmentation(
    volume_np: np.ndarray,
    args,
    metadata,
    reference_image: sitk.Image,
    output_dir: Path,
    case_id: str,
):
    """Run segmentation and save results."""
    t0 = time.time()
    patch_size = tuple(args.patch_size)

    # Decide which model to use
    use_cmf = args.cmf_model is not None and args.cmf_fine_model is not None

    if use_cmf:
        logger.info("[%s] Using CMF-specialized segmentor ...", case_id)
        cfg = CMFSegmentorConfig(
            device=args.device,
            coarse_patch_size=patch_size,
            fine_patch_size=patch_size,
            apply_mar=(metadata.modality == "CBCT"),
        )
        try:
            segmentor = MandibleSegmentor.from_pretrained(args.cmf_model, args.cmf_fine_model, cfg)
        except FileNotFoundError:
            logger.warning("CMF model files not found; falling back to general segmentor.")
            use_cmf = False

    if not use_cmf:
        logger.info("[%s] Using general bone segmentor ...", case_id)
        seg_cfg = SegmentorConfig(
            device=args.device,
            patch_size=patch_size,
            apply_closing=not args.no_postproc,
        )
        try:
            segmentor = BoneSegmentor.from_pretrained(args.model, seg_cfg)
        except FileNotFoundError:
            logger.info("[%s] Model not found; using untrained segmentor (demo mode).", case_id)
            segmentor = BoneSegmentor.build_new(seg_cfg)

    # Run inference
    if use_cmf:
        result = segmentor.segment(volume_np, metadata.spacing, modality=metadata.modality)
    else:
        result = segmentor.predict(volume_np, return_probabilities=args.save_probabilities)

    elapsed = time.time() - t0
    logger.info("[%s] Segmentation complete in %.1f seconds.", case_id, elapsed)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Integer label map
    label_map = result.get("label_map")
    if label_map is not None:
        seg_path = output_dir / "segmentation.nii.gz"
        save_segmentation_nifti(label_map, reference_image, seg_path)
        logger.info("[%s] Saved label map: %s", case_id, seg_path)

    # Per-class binary masks
    masks_saved = []
    for cls_name, mask in result.items():
        if cls_name in ("label_map", "heatmaps", "probabilities") or not isinstance(mask, np.ndarray):
            continue
        if mask.dtype == bool or mask.dtype == np.uint8:
            p = output_dir / f"mask_{cls_name}.nii.gz"
            save_segmentation_nifti(mask.astype(np.uint8), reference_image, p)
            masks_saved.append(cls_name)

    logger.info("[%s] Saved %d binary masks.", case_id, len(masks_saved))

    # Statistics JSON
    stats = {
        "case_id": case_id,
        "segmentation_time_sec": round(elapsed, 2),
        "volume_shape": list(volume_np.shape),
        "modality": metadata.modality,
        "masks": {
            cls: int(result[cls].sum())
            for cls in masks_saved
        },
    }

    with open(output_dir / "segmentation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return result


def main():
    args = parse_args()

    proc_cfg = PreprocessingConfig(
        target_spacing_mm=args.spacing,
        apply_n4_bias_correction=True,
    )
    pipeline = DICOMPipeline(proc_cfg)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if args.batch:
        # Batch mode: process all subdirectories
        if not input_path.is_dir():
            logger.error("--input must be a directory in batch mode.")
            sys.exit(1)

        subdirs = [d for d in sorted(input_path.iterdir()) if d.is_dir()]
        logger.info("Batch mode: %d cases found.", len(subdirs))

        results_summary = []
        for patient_dir in subdirs:
            case_id = patient_dir.name
            case_output = output_dir / case_id
            try:
                volume_np, preprocessed, meta = load_volume(
                    patient_dir, pipeline, args.spacing, args.modality
                )
                run_segmentation(volume_np, args, meta, preprocessed, case_output, case_id)
                results_summary.append({"case_id": case_id, "status": "success"})
            except Exception as exc:
                logger.error("Case %s failed: %s", case_id, exc)
                results_summary.append({"case_id": case_id, "status": "failed", "error": str(exc)})

        with open(output_dir / "batch_summary.json", "w") as f:
            json.dump(results_summary, f, indent=2)
        logger.info("Batch complete. Summary: %s/batch_summary.json", output_dir)

    else:
        # Single case mode
        case_id = input_path.stem.replace(".nii", "")
        try:
            volume_np, preprocessed, meta = load_volume(
                input_path, pipeline, args.spacing, args.modality
            )
            run_segmentation(volume_np, args, meta, preprocessed, output_dir, case_id)
            logger.info("Done. Output: %s", output_dir)
        except Exception as exc:
            logger.error("Segmentation failed: %s", exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
