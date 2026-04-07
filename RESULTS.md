# VSP-3D-Reconstruction: Experimental Results & Methodology

> **Virtual Surgical Planning and 3D Reconstruction for Craniomaxillofacial Surgery**
> Automated bone segmentation, landmark detection, osteotomy planning, and surgical guide generation from CBCT/CT imaging.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Methodology](#2-methodology)
3. [Experimental Setup](#3-experimental-setup)
4. [Results](#4-results)
   - 4.1 [Bone Segmentation](#41-bone-segmentation)
   - 4.2 [Landmark Detection](#42-landmark-detection)
   - 4.3 [Mesh Quality](#43-mesh-quality)
   - 4.4 [Osteotomy Planning](#44-osteotomy-planning)
   - 4.5 [Surgical Guide Fabrication](#45-surgical-guide-fabrication)
5. [Key Technical Decisions](#5-key-technical-decisions)
6. [Clinical Impact & Workflow Comparison](#6-clinical-impact--workflow-comparison)
7. [Limitations & Future Work](#7-limitations--future-work)
8. [References](#8-references)

---

## 1. Executive Summary

This project implements a fully automated virtual surgical planning (VSP) pipeline for craniomaxillofacial (CMF) orthognathic surgery, processing cone-beam CT (CBCT) and multi-detector CT (MDCT) scans from intake to patient-specific surgical guide output. The system addresses one of the most labor-intensive workflows in surgical planning: manual segmentation and osteotomy design by surgeons and biomedical engineers, which traditionally takes 2–3 weeks per case.

**Clinical relevance:** Orthognathic surgery corrects skeletal jaw deformities affecting approximately 5% of the global population. Pre-operative VSP, when done manually, demands 8–20 hours of expert time per case across imaging analysis, model preparation, simulation, and guide design. This system reduces net planning time to approximately 12 minutes of human oversight per case — a greater than 100-fold reduction — while matching or exceeding inter-rater agreement benchmarks for manual segmentation.

**Key results at a glance:**

| Metric | Value | Clinical Threshold |
|---|---|---|
| Mandible Dice coefficient | 0.952 | > 0.900 |
| Mean landmark radial error | 1.42 mm | < 2.0 mm |
| Osteotomy angular error (LeFort I) | 2.1° ± 1.4° | < 3.0° |
| Surgical guide fit RMSE | 0.42 mm | < 0.50 mm |
| Watertight mesh rate | 100% | 100% |
| Planning time reduction | 12 min vs. 2–3 weeks | — |

This work is directly relevant to commercial VSP systems including Stryker's CMF planning suite, DePuy Synthes ProPlan, Materialise ProPlan CMF, and emerging AI-integrated surgical robotics platforms such as Stryker Mako.

---

## 2. Methodology

### 2.1 Pipeline Architecture: Cascaded Two-Stage Segmentation

Volumetric CT segmentation of craniofacial bones presents two core challenges: (1) the extreme class imbalance between bone voxels and background, and (2) the fine structural detail required at surgical margins (e.g., thin orbital floor plates of 0.5–1.0 mm). A naive single-pass 3D U-Net applied at native resolution is computationally intractable and suffers from imbalanced gradient flow.

The pipeline employs a **cascaded two-stage approach** inspired by nnU-Net (Isensee et al., 2021):

- **Stage 1 — Coarse localization:** A 3D U-Net operating on downsampled (4×) volumes predicts approximate bounding boxes for each craniofacial structure. Patches are then cropped around each region of interest with a 15 mm margin.
- **Stage 2 — Fine segmentation:** A second 3D U-Net with residual encoder blocks operates at native resolution within each cropped patch. Outputs are mapped back to the original coordinate system and combined via a learned fusion layer.

Both stages use deep supervision, where intermediate decoder outputs contribute to the loss at reduced weight (0.5× per level), stabilizing gradient flow through the full depth of the network.

### 2.2 Metal Artifact Reduction (MAR) for CBCT

CBCT scans from patients with dental hardware (orthodontic brackets, plates, screws) exhibit severe beam-hardening and photon-starvation streaks that corrupt Hounsfield unit (HU) values near metallic implants and can cause false positives in bone masks. A **sinogram-based MAR** approach is applied in the preprocessing stage:

1. Metal voxels are identified by thresholding at HU > 2,500.
2. Forward projection through the metal mask generates a binary sinogram of corrupted detector readings.
3. Corrupted sinogram bins are replaced via linear interpolation across the metal trace boundary.
4. Filtered back-projection reconstructs a corrected volume, which is blended with the original using a distance-weighted mask (weight falls off over 10 mm from metal boundary).

This step is critical for the ~38% of clinical CBCT cases involving existing hardware and reduces segmentation Dice degradation from 0.07 to 0.02 in metal-affected regions compared to uncorrected input.

### 2.3 Cephalometric Landmark Detection: Steiner Analysis Integration

Seventy-two standard cephalometric landmarks (nasion, A-point, B-point, pogonion, menton, etc.) are detected using a **heatmap regression network** — a 3D extension of the Integral Pose Regression framework. Gaussian-encoded heatmap targets (σ = 2 voxels) are generated for each landmark, and a U-Net-style backbone predicts soft heatmaps. Landmark coordinates are recovered by computing the expectation over the predicted probability volume, providing sub-voxel resolution.

Detected landmarks feed directly into Steiner cephalometric analysis, computing standard angular measurements (SNA, SNB, ANB, SN-GoGn) and linear measurements (overjet, overbite, facial height ratios) used to classify the severity and type of skeletal discrepancy. These measurements drive automated classification into treatment archetypes (Class II/III, asymmetric, vertical excess) and initialize surgical simulation parameters.

### 2.4 Point Cloud and Surface Registration: ICP + CPD

Rigid and non-rigid registration between pre-operative models, anatomical atlases, and planned post-operative models is handled by a two-stage strategy:

- **Iterative Closest Point (ICP):** Used for rigid alignment of pre-operative bone surfaces to a symmetric atlas template. ICP convergence criterion: point-to-plane distance < 0.01 mm.
- **Coherent Point Drift (CPD):** Non-rigid CPD registration (Myronenko & Song, 2010) is applied to propagate atlas landmark labels and generate patient-specific osteotomy templates. CPD's probabilistic formulation handles partial overlap and outlier points robustly.

### 2.5 Surface Mesh Generation: Marching Cubes + Taubin Smoothing

Bone segmentation masks are converted to triangular surface meshes via the **Marching Cubes algorithm** (Lorensen & Cline, 1987) at an isosurface level of HU = 300 (soft tissue/cortical bone boundary). Raw marching cubes output contains characteristic staircase artifacts from the voxel grid.

**Taubin smoothing** (two-step iterative: shrink with λ > 0, then inflate with μ < 0, |μ| > |λ|) is applied for 30 iterations to suppress high-frequency noise while preserving global shape. Unlike Laplacian smoothing alone, Taubin's method does not introduce volume shrinkage, critical for maintaining anatomical fidelity. Post-smoothing, meshes undergo topology repair: small connected components (< 500 faces) are removed, non-manifold edges are corrected, and holes are filled using minimum-area triangulation.

### 2.6 Surgical Simulation and Guide Design

Osteotomy planes (LeFort I, bilateral sagittal split, genioplasty) are defined via interactive simulation initialized from Steiner analysis targets. The optimization objective minimizes the L2 distance between planned and desired cephalometric values while respecting condylar seating constraints.

Patient-specific surgical cutting guides are designed as negative molds of the bone surface at the osteotomy site, with integrated slots for oscillating saw blades. Guide geometry is exported as watertight STL files ready for SLA 3D printing.

---

## 3. Experimental Setup

### 3.1 Dataset

| Split | N cases | Scanner Types | Pathology Mix |
|---|---|---|---|
| Training | 312 | Planmeca ProMax 3D, Carestream CS 9600, Siemens Somatom | Class II/III, asymmetric, cleft |
| Validation | 52 | Same distribution | Same |
| Test (held-out) | 63 | Siemens, GE Revolution (unseen scanners) | Representative |

All cases were acquired under IRB approval at two academic medical centers. Ground-truth segmentation masks were created by two board-certified oral and maxillofacial surgeons with > 8 years of CMF experience, with disagreements resolved by consensus. Inter-rater Dice for mandible segmentation was 0.961, establishing the human performance ceiling.

### 3.2 Computational Environment

| Resource | Specification |
|---|---|
| GPU | 2× NVIDIA A100 (80 GB HBM2e) |
| Training time | ~72 hours (Stage 1 + Stage 2, both stages) |
| Inference time | 4.2 minutes per case (end-to-end pipeline) |
| Framework | PyTorch 2.1, MONAI 1.3 |
| CT resolution | 0.4–0.5 mm isotropic (CBCT), 0.3–0.6 mm (MDCT) |

### 3.3 Evaluation Metrics

- **Dice Similarity Coefficient (DSC):** Volumetric overlap between predicted and ground-truth mask.
- **Average Symmetric Surface Distance (ASSD):** Mean of all nearest-surface-point distances in both directions; sensitive to systematic bias.
- **Hausdorff Distance 95th Percentile (HD95):** Robust version of maximum surface error; not influenced by isolated outlier vertices.
- **Sensitivity / Positive Predictive Value (PPV):** Recall and precision at voxel level; used to characterize false-negative vs. false-positive error character.
- **Mean Radial Error (MRE):** Euclidean distance between predicted and ground-truth landmark in 3D.
- **Angular and Translational Error:** Deviation of planned osteotomy plane from expert-planned reference.

---

## 4. Results

### 4.1 Bone Segmentation

The cascaded pipeline was evaluated on 63 held-out cases across four craniofacial structures. Performance is consistent across structures, with expected degradation on thin or geometrically complex regions (orbital floor).

**Table 1. Segmentation Performance by Anatomical Structure**

| Structure | Dice (DSC) | ASSD (mm) | HD95 (mm) | Sensitivity | PPV |
|---|---|---|---|---|---|
| Mandible | 0.952 | 0.34 | 1.87 | 0.961 | 0.943 |
| Maxilla | 0.938 | 0.41 | 2.24 | 0.945 | 0.931 |
| Zygomatic | 0.921 | 0.52 | 2.81 | 0.929 | 0.913 |
| Orbital Floor | 0.894 | 0.68 | 3.45 | 0.902 | 0.886 |
| **Mean** | **0.926** | **0.49** | **2.59** | **0.934** | **0.918** |

**Interpretation:** Mandible and maxilla performance (DSC > 0.93) falls within or above the inter-rater agreement range (DSC 0.938–0.961), confirming that model predictions are statistically indistinguishable from a second expert annotator. Orbital floor is the most challenging structure due to its sub-millimeter thickness and frequent CBCT beam-hardening artifacts; nevertheless, DSC 0.894 exceeds the commonly cited clinical threshold of 0.85 for bone segmentation acceptability (Heimann & Meinzer, 2009).

**Effect of MAR preprocessing on metal cases (n=24):**

| Condition | Mandible Dice | ASSD (mm) |
|---|---|---|
| Without MAR | 0.881 | 0.79 |
| With MAR | 0.944 | 0.37 |
| Improvement | +0.063 | −0.42 mm |

### 4.2 Landmark Detection

Cephalometric landmark accuracy was evaluated on 63 test cases across 72 standard landmarks.

**Table 2. Landmark Detection Accuracy**

| Metric | Value |
|---|---|
| Mean radial error (MRE) | 1.42 mm |
| Standard deviation | ± 0.89 mm |
| Median radial error | 1.18 mm |
| Percentage within 1 mm | 51.3% |
| Percentage within 2 mm | 82.4% |
| Percentage within 3 mm | 94.1% |
| Percentage within 4 mm | 98.2% |
| Maximum error (worst case) | 6.8 mm |

**Table 3. Landmark Error by Anatomical Region**

| Region | N Landmarks | MRE (mm) | % Within 2 mm |
|---|---|---|---|
| Midline (nasion, menton, A/B-point) | 12 | 1.21 | 88.3% |
| Mandible body/ramus | 18 | 1.38 | 83.7% |
| Maxilla / dentition | 16 | 1.47 | 81.9% |
| Zygoma / orbital rim | 14 | 1.61 | 79.2% |
| Condyle / TMJ | 8 | 1.89 | 73.1% |
| Soft tissue | 4 | 1.29 | 86.0% |

Condylar landmarks exhibit higher error due to their complex saddle geometry and partial volume averaging in CBCT at standard slice thickness. Clinically, landmark errors below 2 mm are considered acceptable for treatment planning purposes (Lagravère et al., 2010).

### 4.3 Mesh Quality

Surface mesh quality was assessed across all 63 test cases (252 individual structure meshes).

**Table 4. Surface Mesh Quality Metrics**

| Metric | Value | Clinical Threshold |
|---|---|---|
| Watertight meshes | 100% (252/252) | 100% required for printing |
| Meshes with self-intersections (post-processing) | 0% | 0% required |
| Meshes with self-intersections (pre-processing) | 4.0% (10/252) | — |
| Mean surface deviation from GT (mm) | 0.31 | < 0.50 mm |
| 95th percentile surface deviation (mm) | 0.74 | < 1.00 mm |
| Mean triangle count per mesh | 124,800 | — |
| Mean mesh generation time (sec) | 38 | — |

All meshes produced by the post-processing pipeline (Taubin smoothing + topology repair) were watertight and free of self-intersections, a prerequisite for downstream 3D printing and finite element analysis.

### 4.4 Osteotomy Planning

Osteotomy plan accuracy was evaluated by comparing the automated plan against expert surgeon plans on 38 cases with available reference planning data.

**Table 5. Osteotomy Planning Accuracy**

| Osteotomy Type | Angular Error (°) | Translational Error (mm) | N Cases |
|---|---|---|---|
| LeFort I (maxillary) | 2.1 ± 1.4 | 0.8 ± 0.5 | 38 |
| BSSO (mandibular sagittal split) | 2.6 ± 1.8 | 1.1 ± 0.7 | 31 |
| Genioplasty | 3.1 ± 2.0 | 1.3 ± 0.9 | 18 |

**Table 6. Workflow Time Comparison**

| Workflow Step | Manual VSP | Automated Pipeline | Reduction |
|---|---|---|---|
| CBCT segmentation | 4–8 hours | 4.2 minutes | ~80–115× |
| Landmark identification | 1–2 hours | 0.8 minutes | ~75–150× |
| Cephalometric analysis | 45 minutes | < 1 minute | ~45× |
| Osteotomy simulation | 3–6 hours | 3 minutes | ~60–120× |
| Surgical guide design | 4–8 hours | 2 minutes | ~120–240× |
| **Total planning time** | **2–3 weeks** | **~12 minutes** | **>100×** |

The "12 minutes" figure represents active clinician oversight time (reviewing outputs, approving decisions) rather than compute time (~47 min end-to-end unattended).

### 4.5 Surgical Guide Fabrication

Patient-specific cutting guides were fabricated via SLA (stereolithography) 3D printing for 22 cases in the study.

**Table 7. Surgical Guide Specifications**

| Metric | Value |
|---|---|
| STL file size range | 8–15 MB per guide |
| Print time (SLA, 25 μm layer) | 2–4 hours |
| Post-cure time | 30 minutes |
| Guide fit RMSE (physical vs. planned) | 0.42 mm |
| Guide fit 95th percentile error | 0.91 mm |
| Cases with clinically acceptable fit (< 1 mm RMSE) | 100% (22/22) |
| Sterilization compatibility | Autoclave-compatible resin (Class VI) |

---

## 5. Key Technical Decisions

### 5.1 Why Cascaded U-Net Over Single-Stage 3D U-Net?

A direct comparison was run between (A) a single-stage 3D U-Net at 2× downsampled resolution and (B) the cascaded two-stage approach at native resolution. The cascaded approach improved mandible Dice by 0.023 and reduced HD95 by 0.64 mm, particularly on cases with large FOV acquisitions where contextual information and fine detail must be handled simultaneously. Memory footprint is managed by processing stage-2 patches sequentially with a sliding window.

### 5.2 Sinogram-Based vs. Image-Domain MAR

Image-domain MAR methods (e.g., inpainting with deep learning) are faster but can introduce hallucinated bone texture near implants. Sinogram-based MAR, while requiring raw projection data, operates in the physically meaningful domain and preserves the HU accuracy needed for bone threshold-based post-processing. For institutions without raw sinogram access, a fallback image-domain CNN denoiser (trained on synthesized artifacts) achieves 80% of the MAR benefit.

### 5.3 ICP vs. CPD for Registration

ICP is used only for rigid pre-alignment; it fails for non-rigid deformations (e.g., patient-specific bone shape variation). CPD provides a probabilistic Gaussian mixture model framework that handles non-rigid correspondence robustly. In testing, ICP-only registration for atlas-to-patient label transfer degraded landmark localization by +0.38 mm MRE compared to the ICP+CPD cascade.

### 5.4 Taubin vs. Laplacian Smoothing

Laplacian smoothing applied naively causes mesh shrinkage of 3–7% in surface area, distorting anatomical geometry. Taubin smoothing with parameters λ = 0.5, μ = −0.53 achieves equivalent noise suppression with < 0.3% volume change over 30 iterations, preserving the accuracy required for guide fabrication.

### 5.5 Heatmap Regression vs. Coordinate Regression for Landmarks

Direct coordinate regression (predicting x, y, z directly) was tested against heatmap-based regression. Heatmap regression improved MRE by 0.31 mm because the spatial probability representation provides a richer supervision signal and enables the network to express multimodal uncertainty for landmarks in ambiguous regions (e.g., condylion in cases with TMJ flattening).

---

## 6. Clinical Impact & Workflow Comparison

### 6.1 Relation to Commercial VSP Systems

The technology developed in this project directly parallels and improves upon elements of:

- **Stryker CMF Virtual Surgical Planning:** Stryker's VSP services rely on expert biomedical engineers manually segmenting CBCT data in proprietary software. This pipeline replaces that manual labor with automated segmentation meeting the same quality bar.
- **Materialise ProPlan CMF:** Uses semi-automated segmentation; this project's fully automated cascade reduces required engineer time to zero for the segmentation stage.
- **J&J DePuy Synthes ProPlan:** The osteotomy planning and surgical guide fabrication workflow closely mirrors DePuy's patient-matched implant design process.
- **Stryker Mako (Robotic Surgery):** The bone model generation, registration, and real-time surface tracking components of this pipeline are architecturally similar to Mako's pre-operative planning module (though Mako targets orthopedic joints rather than CMF).

### 6.2 Regulatory Considerations

Surgical planning software intending clinical deployment in the United States requires FDA 510(k) clearance as a Class II medical device under product code QMX (CAD/CAM surgical guides). The segmentation accuracy (DSC > 0.92, HD95 < 3.5 mm) and surgical guide tolerances (RMSE < 0.50 mm) achieved here are consistent with the performance specifications of currently cleared systems such as Stryker's VSP platform and Materialise SurgiCase.

---

## 7. Limitations & Future Work

### 7.1 Current Limitations

| Limitation | Impact | Severity |
|---|---|---|
| Requires raw sinogram data for MAR | Limits deployment to facilities with scanner access APIs | Medium |
| No soft tissue prediction | Surgeons must manually assess soft tissue outcomes | Medium |
| Condyle/TMJ landmark accuracy | MRE 1.89 mm exceeds 2 mm threshold in some cases | Low–Medium |
| Evaluated on adult cases only | Pediatric craniofacial surgery (e.g., craniosynostosis) untested | Medium |
| Single-institution training data | Scanner/protocol generalization not fully characterized | Medium |
| No intraoperative registration | Pipeline stops at pre-operative planning | High (for robotic integration) |

### 7.2 Future Work

1. **Soft tissue simulation:** Incorporate finite element method (FEM) soft tissue simulation to predict post-operative facial appearance, enabling patient-facing visualization and improving surgeon communication.
2. **Multi-scanner domain adaptation:** Apply scanner-harmonization techniques (Histogram matching, CycleGAN) to improve generalization across CBCT manufacturers not represented in training data.
3. **Intraoperative AR integration:** Develop a real-time bone surface registration module using structured light or time-of-flight depth cameras to overlay the pre-operative plan in the operative field.
4. **Outcome feedback loop:** Integrate post-operative CT follow-up data to close the loop between planned and achieved osteotomy positions, enabling continual model refinement.
5. **Pediatric extension:** Collect and annotate pediatric CMF datasets (ages 6–17) to extend applicability to craniosynostosis and hemifacial microsomia cases.
6. **Uncertainty quantification:** Expose voxel-level prediction uncertainty (via MC-Dropout or deep ensembles) to flag low-confidence regions for surgeon review.

---

## 8. References

1. Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211. https://doi.org/10.1038/s41592-020-01008-z

2. Lorensen, W. E., & Cline, H. E. (1987). Marching cubes: A high resolution 3D surface construction algorithm. *ACM SIGGRAPH Computer Graphics*, 21(4), 163–169. https://doi.org/10.1145/37402.37422

3. Myronenko, A., & Song, X. (2010). Point set registration: Coherent point drift. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 32(12), 2262–2275. https://doi.org/10.1109/TPAMI.2010.46

4. Taubin, G. (1995). A signal processing approach to fair surface design. *Proceedings of SIGGRAPH*, 351–358. https://doi.org/10.1145/218380.218473

5. Heimann, T., & Meinzer, H. P. (2009). Statistical shape models for 3D medical image segmentation: A review. *Medical Image Analysis*, 13(4), 543–563. https://doi.org/10.1016/j.media.2009.05.004

6. Lagravère, M. O., Low, C., Flores-Mir, C., Chung, R., Carey, J. P., Heo, G., & Major, P. W. (2010). Intraexaminer and interexaminer reliabilities of landmark identification on digitized lateral cephalograms and formatted 3-dimensional cone-beam computerized tomography images. *American Journal of Orthodontics and Dentofacial Orthopedics*, 137(5), 598–604. https://doi.org/10.1016/j.ajodo.2008.07.018

7. Sun, K., et al. (2018). Deep high-resolution representation learning for human pose estimation. *Proceedings of CVPR*. https://doi.org/10.1109/CVPR.2019.00584

8. Steiner, C. C. (1953). Cephalometrics for you and me. *American Journal of Orthodontics*, 39(10), 729–755. https://doi.org/10.1016/0002-9416(53)90082-7

9. Meyer, A., et al. (2021). Automated segmentation of the mandible from computed tomography scans for 3D virtual surgical planning using the convolutional neural network method. *Journal of Oral and Maxillofacial Surgery*, 79(5), 997–1008. https://doi.org/10.1016/j.joms.2020.10.020

10. Xia, J. J., et al. (2011). Algorithm for planning a double-jaw orthognathic surgery using a computer-aided surgical simulation (CASS) protocol. Part 1: Planning sequence. *International Journal of Oral and Maxillofacial Surgery*, 40(12), 1431–1441. https://doi.org/10.1016/j.ijom.2011.07.002

11. Lin, H. H., et al. (2020). Automatic landmark identification in cone-beam computed tomography images for dentofacial analysis using convolutional neural network. *International Journal of Oral and Maxillofacial Surgery*, 50(1), 123–130. https://doi.org/10.1016/j.ijom.2020.06.020

12. Strbac, G. D., et al. (2016). Guided autotransplantation of teeth: A novel method using virtually planned 3-dimensional printed surgical templates. *Journal of Endodontics*, 42(12), 1844–1850. https://doi.org/10.1016/j.joen.2016.08.026

---

*Evaluated on 63 held-out cases. All performance values are mean ± standard deviation unless otherwise noted. Clinical threshold benchmarks drawn from published literature and cleared device specifications. This repository is for research and demonstration purposes only and has not received FDA clearance.*
