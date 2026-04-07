#!/usr/bin/env python3
"""
plan_surgery.py — CLI for AI-assisted surgical planning

Usage:
    python scripts/plan_surgery.py --meshes /outputs/patient_001/meshes/ --procedure orthognathic
    python scripts/plan_surgery.py --meshes /outputs/ --procedure BSSO --volume seg.nii.gz
    python scripts/plan_surgery.py --meshes /outputs/ --landmarks landmarks.json --procedure orbital

Arguments:
    --meshes DIR         Directory containing bone mesh STL files
    --volume PATH        Optional: preprocessed volume NIfTI for landmark detection
    --procedure STR      Surgical procedure: orthognathic | lefort_i | bsso | genioplasty | orbital
    --landmarks PATH     Optional: pre-detected landmarks JSON (skips landmark detection step)
    --landmark-model PATH Landmark detector checkpoint
    --spacing FLOAT FLOAT FLOAT  Voxel spacing (dz dy dx); auto-detected from volume if given
    --output-dir DIR     Output directory [default: same as --meshes]
    --overjet FLOAT      Target overjet in mm [default: 2.5]
    --overbite FLOAT     Target overbite in mm [default: 2.0]
    --no-cutting-guides  Skip cutting guide generation
    --no-symmetry        Skip symmetry analysis
    --device STR         cuda | cpu
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plan_surgery")


def parse_args():
    parser = argparse.ArgumentParser(
        description="VSP-3D: AI-assisted surgical planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--meshes", required=True, help="Directory with STL mesh files")
    parser.add_argument("--volume", default=None, help="Preprocessed volume NIfTI")
    parser.add_argument("--procedure", default="orthognathic",
                        choices=["orthognathic", "lefort_i", "bsso", "genioplasty", "orbital"])
    parser.add_argument("--landmarks", default=None, help="Pre-detected landmarks JSON")
    parser.add_argument("--landmark-model", default="models/landmark_detector_cmf.pt")
    parser.add_argument("--spacing", nargs=3, type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overjet", type=float, default=2.5)
    parser.add_argument("--overbite", type=float, default=2.0)
    parser.add_argument("--no-cutting-guides", action="store_true")
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-advance-mm", type=float, default=12.0)
    parser.add_argument("--max-setback-mm", type=float, default=10.0)
    return parser.parse_args()


def load_meshes(mesh_dir: Path) -> Dict:
    """Load all STL/OBJ files in a directory as trimesh objects."""
    try:
        import trimesh
    except ImportError:
        logger.error("trimesh is required: pip install trimesh")
        sys.exit(1)

    mesh_dict = {}
    patterns = ["*.stl", "*.STL", "*.obj", "*.OBJ"]
    for pattern in patterns:
        for f in sorted(mesh_dir.glob(pattern)):
            label = f.stem
            try:
                mesh = trimesh.load(str(f))
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                mesh_dict[label] = mesh
                logger.info("Loaded mesh: %s (%d faces)", label, len(mesh.faces))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", f.name, exc)

    return mesh_dict


def load_landmarks(landmarks_path: Path) -> Dict[str, np.ndarray]:
    """Load landmarks from JSON file."""
    with open(landmarks_path) as f:
        lm_raw = json.load(f)
    return {name: np.array(coords) for name, coords in lm_raw.items()}


def detect_landmarks(volume_path: Path, args, spacing=None) -> Dict[str, np.ndarray]:
    """Run landmark detection on a CT/CBCT volume."""
    import SimpleITK as sitk
    from src.landmark_detector import LandmarkDetector, LandmarkDetectorConfig

    logger.info("Running landmark detection on: %s", volume_path)
    image = sitk.ReadImage(str(volume_path))
    arr = sitk.GetArrayFromImage(image).astype("float32")

    if spacing is None:
        sp = image.GetSpacing()
        spacing = (float(sp[2]), float(sp[1]), float(sp[0]))

    cfg = LandmarkDetectorConfig(device=args.device)
    try:
        detector = LandmarkDetector.from_pretrained(args.landmark_model, cfg)
    except FileNotFoundError:
        logger.warning("Landmark model not found; using untrained model (positions will be random).")
        detector = LandmarkDetector.build_new(cfg)

    return detector.predict(arr, spacing)


def save_plan_json(plan, output_dir: Path, case_id: str) -> Path:
    """Serialise surgical plan to JSON."""
    plan_data = {
        "case_id": case_id,
        "procedure": plan.procedure,
        "movements": {
            seg: mv.tolist() for seg, mv in plan.movements.items()
        },
        "osteotomy_planes": [
            {
                "name": p.name,
                "normal": p.normal.tolist(),
                "origin": p.origin.tolist(),
                "procedure": p.procedure,
            }
            for p in plan.osteotomy_planes
        ],
        "collision_free": plan.collision_free,
        "symmetry": plan.symmetry_analysis,
        "cephalometrics_pre": plan.cephalometrics_pre,
        "notes": plan.notes,
    }

    path = output_dir / "surgical_plan.json"
    with open(path, "w") as f:
        json.dump(plan_data, f, indent=2)
    logger.info("Surgical plan saved: %s", path)
    return path


def main():
    args = parse_args()

    mesh_dir = Path(args.meshes)
    if not mesh_dir.is_dir():
        logger.error("Mesh directory not found: %s", mesh_dir)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else mesh_dir / "plan"
    output_dir.mkdir(parents=True, exist_ok=True)
    case_id = mesh_dir.parent.name or "case"

    # 1. Load meshes
    logger.info("Loading bone meshes from: %s", mesh_dir)
    mesh_dict = load_meshes(mesh_dir)
    if not mesh_dict:
        logger.error("No mesh files found in %s", mesh_dir)
        sys.exit(1)
    logger.info("Loaded %d mesh structures.", len(mesh_dict))

    # 2. Landmarks
    landmarks = {}
    if args.landmarks:
        logger.info("Loading pre-detected landmarks: %s", args.landmarks)
        landmarks = load_landmarks(Path(args.landmarks))
    elif args.volume:
        spacing = tuple(args.spacing) if args.spacing else None
        landmarks = detect_landmarks(Path(args.volume), args, spacing)
    else:
        logger.warning("No landmarks or volume provided; planning will use geometry-based estimates.")

    logger.info("Available landmarks: %d", len(landmarks))

    # 3. Cephalometric analysis
    if landmarks:
        from src.planning.landmark_detector import CephalometricAnalysis
        ceph = CephalometricAnalysis()
        ceph_results = ceph.run_full_analysis(landmarks)
        logger.info("Cephalometric analysis:")
        for key, val in ceph_results.items():
            logger.info("  %s = %.2f", key, val)

        ceph_path = output_dir / "cephalometrics_pre.json"
        with open(ceph_path, "w") as f:
            json.dump({k: round(v, 3) for k, v in ceph_results.items()}, f, indent=2)
    else:
        ceph_results = {}

    # 4. Surgical planning
    from src.planning.osteotomy_planner import OsteotomyPlanner

    # Infer spacing from volume or default
    spacing = tuple(args.spacing) if args.spacing else (0.5, 0.5, 0.5)

    planner = OsteotomyPlanner(mesh_dict, landmarks, spacing=spacing)

    t0 = time.time()
    if args.procedure in ("orthognathic", "lefort_i", "bsso"):
        logger.info("Planning %s surgery ...", args.procedure)
        plan = planner.plan_orthognathic(
            target_overjet=args.overjet,
            target_overbite=args.overbite,
            midline_correction=True,
            max_maxillary_advance_mm=args.max_advance_mm,
            max_mandibular_setback_mm=args.max_setback_mm,
        )
    elif args.procedure == "genioplasty":
        logger.info("Planning genioplasty ...")
        from src.planning.osteotomy_planner import SurgicalPlan, BoneSegment
        geni_plane = planner.estimate_genioplasty_plane()
        plan = SurgicalPlan(
            case_id=case_id,
            procedure="genioplasty",
            original_segments={},
            planned_segments={},
            osteotomy_planes=[geni_plane],
            movements={"chin": np.array([0, 0, 5.0, 0, 0, 0])},  # 5mm advance
            landmarks_pre=landmarks,
        )
    else:
        logger.warning("Procedure '%s' not fully implemented; generating basic plan.", args.procedure)
        from src.planning.osteotomy_planner import SurgicalPlan
        plan = SurgicalPlan(
            case_id=case_id,
            procedure=args.procedure,
            original_segments={},
            planned_segments={},
            osteotomy_planes=[],
            movements={},
            landmarks_pre=landmarks,
        )

    plan.cephalometrics_pre = ceph_results
    elapsed = time.time() - t0
    logger.info("Planning complete in %.2f seconds.", elapsed)

    # Report
    logger.info("Plan summary:")
    for seg, mv in plan.movements.items():
        logger.info("  %s: tx=%.1f ty=%.1f tz=%.1f rx=%.1f ry=%.1f rz=%.1f",
                    seg, *mv[:6])
    if plan.symmetry_analysis:
        logger.info("  Symmetry index: %.3f | Midline deviation: %.2f mm",
                    plan.symmetry_analysis.get("facial_symmetry_index", 0),
                    plan.symmetry_analysis.get("midline_deviation_mm", 0))
    logger.info("  Collision free: %s", plan.collision_free)

    # 5. Save plan
    plan_path = save_plan_json(plan, output_dir, case_id)

    # 6. Cutting guides
    if not args.no_cutting_guides and plan.osteotomy_planes:
        logger.info("Generating cutting guides ...")
        try:
            guides = planner.generate_cutting_guides(plan)
            guide_dir = output_dir / "cutting_guides"
            guide_dir.mkdir(exist_ok=True)

            import trimesh
            for guide_name, guide_mesh in guides.items():
                guide_path = guide_dir / f"{guide_name}_guide.stl"
                guide_mesh.export(str(guide_path))
                logger.info("  Guide: %s", guide_path.name)

            logger.info("Generated %d cutting guides.", len(guides))
        except Exception as exc:
            logger.warning("Cutting guide generation failed: %s", exc)

    # 7. Soft tissue prediction
    if plan.movements:
        from src.planning.osteotomy_planner import OsteotomyPlanner as OP
        soft_tissue = OP.predict_soft_tissue_changes(plan.movements)
        st_path = output_dir / "soft_tissue_prediction.json"
        with open(st_path, "w") as f:
            json.dump({k: v.tolist() for k, v in soft_tissue.items()}, f, indent=2)
        logger.info("Soft tissue predictions: %s", st_path)

    logger.info("=" * 60)
    logger.info("Surgical planning complete!")
    logger.info("  Plan:    %s", plan_path)
    logger.info("  Outputs: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
