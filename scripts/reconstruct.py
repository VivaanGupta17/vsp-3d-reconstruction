#!/usr/bin/env python3
"""
reconstruct.py — CLI for 3D mesh reconstruction from segmentation

Usage:
    python scripts/reconstruct.py --segmentation /outputs/patient_001/segmentation.nii.gz
    python scripts/reconstruct.py --segmentation /outputs/ --batch
    python scripts/reconstruct.py --segmentation seg.nii.gz --smoothing taubin --reduction 0.7

Arguments:
    --segmentation    Path to segmentation .nii.gz or directory of segmentation files
    --output-dir      Output directory for meshes [default: same as segmentation dir]
    --smoothing       Smoothing method: taubin | laplacian | none [default: taubin]
    --reduction       Mesh decimation fraction 0–1 [default: 0.7]
    --max-edge-mm     Max edge length for 3D printing [default: 2.0]
    --quality-check   Run full quality QA report
    --check-si        Check for self-intersections (expensive)
    --batch           Process all .nii.gz files in --segmentation directory
    --export-formats  Comma-separated list: stl,obj [default: stl]
    --class-ids       Comma-separated class IDs to reconstruct [default: all foreground]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reconstruction.mesh_generator import MeshGenerator, MeshGeneratorConfig
from src.segmentation.bone_segmentor import CMF_CLASSES if False else None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reconstruct")

# Default class name mapping from CMF segmentation
DEFAULT_CLASS_NAMES = {
    1: "cortical_bone",
    2: "cancellous_bone",
    3: "teeth",
    4: "soft_tissue",
}

CMF_CLASS_NAMES = {
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="VSP-3D: 3D mesh reconstruction from segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoothing", choices=["taubin", "laplacian", "none"], default="taubin")
    parser.add_argument("--reduction", type=float, default=0.7)
    parser.add_argument("--max-edge-mm", type=float, default=2.0)
    parser.add_argument("--quality-check", action="store_true")
    parser.add_argument("--check-si", action="store_true", help="Check self-intersections (slow)")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--export-formats", default="stl")
    parser.add_argument("--class-ids", default=None, help="e.g. '1,2,3,7,8' for selected classes")
    parser.add_argument("--min-voxels", type=int, default=1000, help="Skip classes with fewer voxels")
    return parser.parse_args()


def reconstruct_from_labelmap(
    label_map: np.ndarray,
    spacing: Tuple[float, float, float],
    class_names: Dict[int, str],
    config: MeshGeneratorConfig,
    output_dir: Path,
    formats: List[str],
    class_ids: Optional[List[int]] = None,
    min_voxels: int = 1000,
    run_qa: bool = False,
) -> Dict:
    """Reconstruct meshes for all (or selected) classes in a label map."""
    gen = MeshGenerator(config)
    results = {}

    for cls_id, cls_name in class_names.items():
        if class_ids and cls_id not in class_ids:
            continue

        mask = (label_map == cls_id)
        n_vox = int(mask.sum())
        if n_vox < min_voxels:
            logger.info("Skipping class %d (%s): only %d voxels.", cls_id, cls_name, n_vox)
            continue

        logger.info("Reconstructing class %d: %s (%d voxels) ...", cls_id, cls_name, n_vox)
        t0 = time.time()

        try:
            mesh = gen.generate(mask, voxel_spacing=spacing, label=cls_name)
            elapsed = time.time() - t0
            logger.info("  → %d vertices, %d faces in %.1fs", len(mesh.vertices), len(mesh.faces), elapsed)

            # Export
            for fmt in formats:
                out_path = output_dir / f"{cls_name}.{fmt}"
                if fmt == "stl":
                    gen.export_stl(mesh, out_path)
                elif fmt == "obj":
                    gen.export_obj(mesh, out_path)

            # Quality check
            qa_report = None
            if run_qa:
                qa_report = gen.quality_check(mesh, voxel_spacing=spacing,
                                               check_self_intersections=config.check_self_intersections)
                logger.info("  QA: watertight=%s | ASSD=N/A | min_edge=%.2fmm",
                            qa_report.is_watertight, qa_report.mean_edge_length_mm)

            results[cls_name] = {
                "class_id": cls_id,
                "n_voxels": n_vox,
                "n_vertices": len(mesh.vertices),
                "n_faces": len(mesh.faces),
                "watertight": mesh.is_watertight,
                "reconstruction_time_sec": round(elapsed, 2),
                "qa": {
                    "is_watertight": qa_report.is_watertight,
                    "n_degenerate": qa_report.n_degenerate_faces,
                    "volume_mm3": qa_report.volume_mm3,
                    "surface_area_mm2": qa_report.surface_area_mm2,
                } if qa_report else None,
            }

        except Exception as exc:
            logger.error("  Failed to reconstruct %s: %s", cls_name, exc)
            results[cls_name] = {"class_id": cls_id, "error": str(exc)}

    return results


def process_one(
    seg_path: Path,
    args,
    class_names: Dict[int, str],
    gen_config: MeshGeneratorConfig,
    formats: List[str],
    class_ids: Optional[List[int]],
) -> Dict:
    """Process a single segmentation NIfTI file."""
    logger.info("Loading segmentation: %s", seg_path)
    sitk_seg = sitk.ReadImage(str(seg_path))
    label_map = sitk.GetArrayFromImage(sitk_seg).astype(np.uint8)
    spacing_xyz = sitk_seg.GetSpacing()
    spacing = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))  # (dz, dy, dx)

    output_dir = Path(args.output_dir) if args.output_dir else seg_path.parent / "meshes"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = reconstruct_from_labelmap(
        label_map=label_map,
        spacing=spacing,
        class_names=class_names,
        config=gen_config,
        output_dir=output_dir,
        formats=formats,
        class_ids=class_ids,
        min_voxels=args.min_voxels,
        run_qa=args.quality_check,
    )

    summary_path = output_dir / "reconstruction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Summary saved: %s", summary_path)

    return results


def main():
    args = parse_args()

    # Build generator config
    gen_config = MeshGeneratorConfig(
        smoothing_method=args.smoothing if args.smoothing != "none" else "none",
        target_reduction=args.reduction,
        max_edge_length_print_mm=args.max_edge_mm,
        check_self_intersections=args.check_si,
    )

    formats = [f.strip() for f in args.export_formats.split(",")]
    class_ids = [int(c) for c in args.class_ids.split(",")] if args.class_ids else None

    # Determine which class name dict to use
    # Heuristic: if n_classes > 5, use CMF names; else use default
    class_names = CMF_CLASS_NAMES  # use CMF by default for surgical planning

    seg_path = Path(args.segmentation)

    if args.batch:
        if not seg_path.is_dir():
            logger.error("--segmentation must be a directory in batch mode.")
            sys.exit(1)

        seg_files = sorted(seg_path.rglob("segmentation.nii.gz"))
        logger.info("Batch mode: %d segmentation files.", len(seg_files))

        all_results = {}
        for sf in seg_files:
            case_id = sf.parent.name
            try:
                # Per-case output dir
                if args.output_dir:
                    case_output = Path(args.output_dir) / case_id
                    orig_output = args.output_dir
                    args.output_dir = str(case_output)
                    result = process_one(sf, args, class_names, gen_config, formats, class_ids)
                    args.output_dir = orig_output
                else:
                    result = process_one(sf, args, class_names, gen_config, formats, class_ids)
                all_results[case_id] = result
            except Exception as exc:
                logger.error("Case %s failed: %s", case_id, exc)
                all_results[case_id] = {"error": str(exc)}

        summary_out = Path(args.output_dir or str(seg_path)) / "batch_reconstruction_summary.json"
        with open(summary_out, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Batch complete. Summary: %s", summary_out)

    else:
        if not seg_path.exists():
            logger.error("Segmentation file not found: %s", seg_path)
            sys.exit(1)
        process_one(seg_path, args, class_names, gen_config, formats, class_ids)
        logger.info("Reconstruction complete.")


if __name__ == "__main__":
    main()
