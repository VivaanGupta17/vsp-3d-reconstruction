#!/usr/bin/env python3
"""
export_stl.py — CLI for final STL export for 3D printing

Prepares the full surgical 3D printing package:
  - Bone models (mandible, maxilla, surgical segments)
  - Cutting guides
  - Drill/screw templates
  - Positioning splints (if occlusion data available)
  - Nesting layout for optimal build plate usage
  - Generates a QA report for print readiness

Usage:
    python scripts/export_stl.py --plan /outputs/patient_001/plan/surgical_plan.json
    python scripts/export_stl.py --meshes /outputs/patient_001/meshes/ --plan plan.json
    python scripts/export_stl.py --meshes /outputs/ --plan plan.json --output stl_package/

Arguments:
    --plan PATH          Surgical plan JSON from plan_surgery.py
    --meshes DIR         Directory with mesh files (overrides plan mesh dir)
    --output DIR         Output STL package directory
    --no-nest            Skip nesting optimisation
    --build-vol X Y Z    3D printer build volume in mm [default: 200 200 200]
    --include-guides     Include cutting guides (default: true)
    --include-models     Include bone models (default: true)
    --target-edge-mm     Max edge length for final mesh [default: 1.5]
    --watertight-only    Skip any mesh that is not watertight
    --report             Generate PDF readiness report
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_stl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="VSP-3D: Export STL package for 3D printing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--plan", default=None, help="Surgical plan JSON")
    parser.add_argument("--meshes", required=True, help="Mesh directory")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-nest", action="store_true")
    parser.add_argument("--build-vol", nargs=3, type=float, default=[200, 200, 200])
    parser.add_argument("--include-guides", action="store_true", default=True)
    parser.add_argument("--include-models", action="store_true", default=True)
    parser.add_argument("--target-edge-mm", type=float, default=1.5)
    parser.add_argument("--watertight-only", action="store_true")
    parser.add_argument("--reduction", type=float, default=0.0, help="Additional decimation for print")
    parser.add_argument("--report", action="store_true", help="Generate QA report")
    return parser.parse_args()


def load_mesh_dict(mesh_dir: Path) -> Dict:
    """Load all STL/OBJ files from directory."""
    try:
        import trimesh
    except ImportError:
        logger.error("trimesh required: pip install trimesh")
        sys.exit(1)

    meshes = {}
    for f in sorted(mesh_dir.glob("*.stl")) + sorted(mesh_dir.glob("*.STL")):
        try:
            mesh = trimesh.load(str(f))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            meshes[f.stem] = mesh
        except Exception as exc:
            logger.warning("Failed to load %s: %s", f.name, exc)

    return meshes


def run_qa(mesh, label: str) -> Dict:
    """Run print readiness QA on a mesh."""
    qa = {
        "label": label,
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.faces),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "surface_area_mm2": round(float(mesh.area), 2),
        "volume_mm3": round(float(abs(mesh.volume)), 2) if mesh.is_watertight else None,
        "bounding_box_mm": [round(float(x), 2) for x in mesh.bounding_box.extents],
        "euler_number": int(mesh.euler_number),
        "print_ready": mesh.is_watertight and mesh.is_winding_consistent,
        "warnings": [],
    }

    if not mesh.is_watertight:
        qa["warnings"].append("Mesh is not watertight — may need repair before printing.")
    if not mesh.is_winding_consistent:
        qa["warnings"].append("Inconsistent face winding — normals may be inverted.")
    if mesh.euler_number != 2:
        qa["warnings"].append(f"Non-manifold topology (Euler={mesh.euler_number}) — check for holes.")

    bbox = mesh.bounding_box.extents
    if any(b > 200 for b in bbox):
        qa["warnings"].append(f"Model may exceed standard build volume: {[round(b,1) for b in bbox]} mm")

    return qa


def repair_mesh(mesh):
    """Attempt basic mesh repair: fill holes, fix normals."""
    import trimesh
    # Fix normals and winding
    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fill_holes(mesh)
    except Exception as exc:
        logger.debug("Mesh repair partial: %s", exc)
    return mesh


def generate_report(
    qa_results: List[Dict],
    output_path: Path,
    plan_data: Optional[Dict] = None,
) -> None:
    """Generate a markdown print readiness report."""
    lines = [
        "# VSP-3D: 3D Printing Readiness Report",
        "",
        f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]

    if plan_data:
        lines += [
            f"**Case ID:** {plan_data.get('case_id', 'N/A')}",
            f"**Procedure:** {plan_data.get('procedure', 'N/A')}",
        ]

    lines += ["", "## Mesh Quality Summary", ""]
    lines.append("| Model | Faces | Watertight | Volume (mm³) | BBox (mm) | Print Ready |")
    lines.append("|-------|-------|-----------|-------------|---------|------------|")

    all_ready = True
    for qa in qa_results:
        bbox_str = " × ".join(str(b) for b in qa.get("bounding_box_mm", []))
        vol = qa.get("volume_mm3", "N/A")
        ready = "✓" if qa["print_ready"] else "✗"
        if not qa["print_ready"]:
            all_ready = False
        lines.append(
            f"| {qa['label']} | {qa['n_faces']:,} | {qa['is_watertight']} | "
            f"{vol} | {bbox_str} | {ready} |"
        )

    lines += [
        "",
        f"**Overall print readiness:** {'PASS ✓' if all_ready else 'REVIEW REQUIRED ✗'}",
        "",
        "## Warnings",
    ]

    for qa in qa_results:
        for warn in qa.get("warnings", []):
            lines.append(f"- **{qa['label']}**: {warn}")

    if not any(qa.get("warnings") for qa in qa_results):
        lines.append("No warnings. All models are print-ready.")

    lines += [
        "",
        "## Manufacturing Notes",
        "",
        "- **Material recommendation:** Photopolymer resin (SLA/DLP) for surgical guides; "
          "Nylon PA12 (SLS) for anatomical models",
        "- **Layer height:** 0.1 mm for surgical guides; 0.2 mm for anatomical models",
        "- **Sterilisation:** Surgical guides must be sterilised per AAMI ST91:2021",
        "- **Biocompatibility:** Ensure ISO 10993 compliant materials for patient-contact parts",
        "- **Dimensional accuracy:** Validate printed guide fit on physical model before clinical use",
        "",
        "---",
        "*This report is generated by VSP-3D and is for research use only. "
          "Not for clinical decision-making without qualified clinical review.*",
    ]

    report_text = "\n".join(lines)
    output_path.write_text(report_text)
    logger.info("Print readiness report: %s", output_path)


def main():
    args = parse_args()

    mesh_dir = Path(args.meshes)
    if not mesh_dir.is_dir():
        logger.error("Mesh directory not found: %s", mesh_dir)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else mesh_dir.parent / "stl_package"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load plan JSON if provided
    plan_data = None
    if args.plan:
        try:
            with open(args.plan) as f:
                plan_data = json.load(f)
            logger.info("Loaded surgical plan: %s", args.plan)
        except Exception as exc:
            logger.warning("Failed to load plan: %s", exc)

    try:
        import trimesh
    except ImportError:
        logger.error("trimesh is required: pip install trimesh")
        sys.exit(1)

    from src.reconstruction.mesh_generator import MeshGenerator, MeshGeneratorConfig

    gen_cfg = MeshGeneratorConfig(
        smoothing_method="none",  # already smoothed in reconstruction step
        target_reduction=args.reduction,
        max_edge_length_print_mm=args.target_edge_mm,
        fix_normals=True,
    )
    gen = MeshGenerator(gen_cfg)

    # Load meshes from mesh directory (models)
    all_meshes = {}
    if args.include_models:
        model_dir = mesh_dir
        all_meshes.update(load_mesh_dict(model_dir))
        logger.info("Loaded %d bone models.", len(all_meshes))

    # Load cutting guides if present
    if args.include_guides:
        guide_dir = mesh_dir.parent / "plan" / "cutting_guides"
        if guide_dir.exists():
            guides = load_mesh_dict(guide_dir)
            all_meshes.update({f"guide_{k}": v for k, v in guides.items()})
            logger.info("Loaded %d cutting guides.", len(guides))
        else:
            logger.info("No cutting guides directory found at %s", guide_dir)

    if not all_meshes:
        logger.error("No meshes loaded. Check --meshes directory.")
        sys.exit(1)

    logger.info("Total meshes to export: %d", len(all_meshes))

    # Process and export each mesh
    qa_results = []
    exported_paths = []
    skipped = []

    for label, mesh in all_meshes.items():
        logger.info("Processing: %s (%d faces, watertight=%s)", label, len(mesh.faces), mesh.is_watertight)

        # Attempt repair
        if not mesh.is_watertight:
            logger.info("  Attempting repair ...")
            mesh = repair_mesh(mesh)

        # Skip non-watertight if requested
        if args.watertight_only and not mesh.is_watertight:
            logger.warning("  Skipping %s: not watertight.", label)
            skipped.append(label)
            continue

        # Additional decimation if requested
        if args.reduction > 0:
            mesh = gen._decimate(mesh)

        # QA
        qa = run_qa(mesh, label)
        qa_results.append(qa)

        if qa["warnings"]:
            for w in qa["warnings"]:
                logger.warning("  ⚠ %s", w)

        # Export STL
        out_path = output_dir / f"{label}.stl"
        gen.export_stl(mesh, out_path)
        exported_paths.append(out_path)

    logger.info("Exported %d STL files.", len(exported_paths))
    if skipped:
        logger.warning("Skipped %d non-watertight meshes: %s", len(skipped), skipped)

    # Nesting layout
    if not args.no_nest and len(all_meshes) > 1:
        logger.info("Creating nesting layout ...")
        try:
            valid_meshes = {k: v for k, v in all_meshes.items() if k not in skipped}
            nested_scene = gen.nest_for_printing(
                valid_meshes,
                build_volume_mm=tuple(args.build_vol),
            )
            nest_path = output_dir / "nested_layout.stl"
            nested_scene.export(str(nest_path))
            logger.info("Nesting layout: %s", nest_path)
        except Exception as exc:
            logger.warning("Nesting failed: %s", exc)

    # QA report
    qa_json_path = output_dir / "print_qa_report.json"
    with open(qa_json_path, "w") as f:
        json.dump(qa_results, f, indent=2)

    if args.report:
        md_path = output_dir / "print_readiness_report.md"
        generate_report(qa_results, md_path, plan_data)

    # Manifest
    manifest = {
        "case_id": plan_data.get("case_id", mesh_dir.parent.name) if plan_data else mesh_dir.parent.name,
        "procedure": plan_data.get("procedure", "N/A") if plan_data else "N/A",
        "exported_files": [str(p.name) for p in exported_paths],
        "skipped": skipped,
        "print_ready_count": sum(1 for qa in qa_results if qa["print_ready"]),
        "total_count": len(qa_results),
    }

    manifest_path = output_dir / "export_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    n_ready = manifest["print_ready_count"]
    n_total = manifest["total_count"]
    logger.info("=" * 60)
    logger.info("STL Export Complete")
    logger.info("  Output:      %s", output_dir)
    logger.info("  Files:       %d exported", len(exported_paths))
    logger.info("  Print Ready: %d/%d", n_ready, n_total)
    logger.info("  QA Report:   %s", qa_json_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
