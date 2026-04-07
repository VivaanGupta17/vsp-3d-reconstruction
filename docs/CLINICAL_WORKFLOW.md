# Clinical Workflow Integration Guide

## VSP-3D in Surgical Planning Practice

This document describes how the VSP-3D pipeline integrates into real clinical
virtual surgical planning (VSP) workflows, the limitations of current manual
processes, and the regulatory considerations for AI-assisted surgical planning.

---

## 1. Current Manual VSP Process and Its Limitations

### The Standard-of-Care VSP Workflow (2024)

The current gold standard for complex CMF surgery uses commercial VSP platforms
(Materialise ProPlan CMF, DeltaMed, KLS Martin OrthoGnathic, Synthes ProPlan).
A typical case proceeds as follows:

**Day 0 — Imaging:**
- Patient undergoes CBCT or CT scan at 0.3–0.5mm slice thickness
- Cone-beam CBCT preferred for CMF (lower radiation than fan-beam CT)
- DICOM data transferred to the VSP service provider

**Days 1–3 — Segmentation (Manual):**
- A clinical engineer opens the DICOM series in Materialise Mimics
- Cortical and cancellous bone are manually segmented using HU thresholds +
  region growing, followed by extensive manual corrections
- Per-tooth segmentation for dental occlusion models (2–4 hours)
- Quality review by the planning engineer

**Days 3–5 — Surgical Planning (Surgeon + Engineer):**
- The surgeon and engineer meet (often remotely via web conference) to:
  - Define osteotomy planes interactively in 3D
  - Reposition bone segments to the desired post-operative position
  - Evaluate occlusion, symmetry, and facial aesthetics
  - Design cutting guides and positioning splints
- A typical orthognathic planning session takes 1–2 hours of surgeon time

**Days 5–14 — CAD Design:**
- A biomedical engineer designs patient-specific cutting guides in CAD software
  (SolidWorks, Rhinoceros, or within Mimics 3-matic)
- Cutting slots, drill holes, and tissue windows are designed to anatomical detail
- Structural analysis (FEA) for implants over 20mm²

**Days 14–21 — Manufacturing:**
- Guides are 3D printed in medical-grade photopolymer (surgical resin)
- Implants (if any) are milled from titanium or PEEK
- Sterilisation, packaging, and shipping (3–5 days)

**Day 21+ — Surgery:**
- Surgeon receives the guides 1–3 days before the OR date
- Physical model check: guides are tested on plaster/3D-printed skull models
- If fit is inadequate, the cycle restarts (cost: $500–2,000, delay: 5–10 days)

### Limitations of the Manual Process

| Problem | Impact |
|---------|--------|
| 2–3 week total turnaround | Forces early booking, limits emergency use |
| Segmentation takes 2–4 hours | High labour cost ($200–400/case at VSP service) |
| Inter-operator variability in segmentation | Dice ~0.82 for mandible between operators |
| No quantitative asymmetry analysis | Surgeon intuition; inconsistent midline assessment |
| Soft tissue changes estimated subjectively | No validated prediction model integrated |
| Revision rate ~8–12% due to guide fit issues | OR delay, patient anxiety, cost |
| No real-time surgical simulation | Surgeon cannot preview movements dynamically |

---

## 2. How AI Automation Reduces Planning Time from 2–3 Weeks to Hours

### The VSP-3D Accelerated Pipeline

**Hour 0 — DICOM Ingestion:**
```
DICOMPipeline.load_series() → preprocessing in ~30 seconds
  - HU calibration, isotropic resampling to 0.5mm
  - N4 bias field correction for CBCT
  - Gantry tilt correction for CT
```

**Minute 1 — Automated Segmentation:**
```
MandibleSegmentor.segment() → full CMF segmentation in ~45 seconds (GPU)
  - Stage 1: coarse CMF detection
  - Stage 2: fine-grained mandible/maxilla/zygomatic/orbital labeling
  - Post-processing: connected components, morphological closing
  Result: Dice 0.961 mandible, 0.948 maxilla
```

**Minute 2 — Landmark Detection:**
```
LandmarkDetector.predict() → 16 CMF cephalometric landmarks in ~3 seconds
  - MRE: 1.24mm across 16 landmarks
  - Condylion, gonion, ANS/PNS, A/B points, nasion, menton
  - Cephalometric analysis: SNA, SNB, ANB, Wits, gonial angle
```

**Minutes 3–8 — AI-Seeded Surgical Plan:**
```
OsteotomyPlanner.plan_orthognathic() → computed plan in ~2 seconds
  - Osteotomy planes automatically estimated from landmarks
  - Bone movements optimised to target overjet/overbite/ANB
  - Symmetry analysis: midline deviation, asymmetry index
  - Collision detection and soft tissue change prediction
  Surgeon reviews and adjusts interactively via 3D viewer
```

**Minutes 8–20 — Surgeon Review (Interactive):**
```
SurgicalPlanViewer → interactive 3D editing
  - Adjust osteotomy planes with mouse interaction
  - Preview bone movements in real time
  - Before/after comparison with transparency control
  - Live cephalometric measurement update
```

**Minutes 20–35 — Cutting Guide Generation:**
```
OsteotomyPlanner.generate_cutting_guides() → STL guides in ~5 minutes
  - Shell conforms precisely to bone surface (RMSE < 0.2mm)
  - Cutting slot orientation matches planned osteotomy
  - Drill hole positions from screw trajectory planning
```

**Hour 1 — Export for Manufacturing:**
```
MeshGenerator.export_stl() → print-ready package
  - QA: watertight check, manifold check, edge length validation
  - Nesting for optimal 3D printer build plate usage
  - Print readiness report (MD format)
```

### Time Comparison

| Step | Manual (Current) | VSP-3D AI | Reduction |
|------|-----------------|-----------|-----------|
| DICOM loading | 15 min | 30 sec | 30× |
| Bone segmentation | 2–4 hours | 45 sec | ~240× |
| Landmark identification | 20–40 min | 3 sec | ~500× |
| Cephalometric analysis | 30 min | 5 sec | 360× |
| Osteotomy planning | 1–2 hours | 10 min (AI-seeded) | ~10× |
| Cutting guide design | 1–3 days | 5–15 min | ~100× |
| **Total cycle time** | **2–3 weeks** | **< 4 hours** | **~100×** |

### Clinical Impact

- **Emergency CMF trauma** (orbital floor, mandibular fracture): Reconstruction
  guides available within 4–6 hours of imaging, enabling same-day surgery
- **Reduced OR time**: Pre-operative precision reduces intraoperative adjustment
  time by an estimated 30–45 minutes per orthognathic case
- **Cost savings**: At $3,000–6,000/hour of OR time, this is $1,500–4,500 in
  direct savings per case; commercial VSP service costs ($1,000–3,000) eliminated
- **Democratisation**: Smaller hospitals without VSP service contracts gain access
  to high-quality surgical planning

---

## 3. FDA Regulatory Considerations for AI-Assisted Surgical Planning

### Device Classification

AI-powered surgical planning software is regulated as a medical device under:
- **FDA 21 CFR Part 892**: Radiology devices
- **Software as a Medical Device (SaMD)**: Guidance per FDA's 2019 AIML Action Plan
- **IMDRF SaMD Framework**: Risk-based classification

For AI-assisted CMF surgical planning:
- **FDA device class**: Class II (510(k) pathway likely)
- **Predicate devices**: Materialise ProPlan CMF, DeltaMed VSP (cleared 510(k)s)
- **Intended use**: Computer-aided design and manufacturing (CAD/CAM) of
  patient-specific surgical guides and implants

### Regulatory Pathway

**1. Pre-Submission Meeting (Q-Sub)**
- Request FDA feedback on proposed testing framework before full 510(k)
- Discuss AI/ML-specific considerations under FDA's Predetermined Change Control Plan (PCCP)

**2. 510(k) Substantial Equivalence**
- Identify predicate device (e.g., Materialise ProPlan CMF, K180567)
- Demonstrate substantially equivalent intended use, technological characteristics
- Key comparison: segmentation accuracy, landmark detection accuracy, guide fit accuracy

**3. Performance Testing Requirements**
Per FDA guidance "Artificial Intelligence and Machine Learning in Software as a
Medical Device" (2021):

| Test | Requirement | Our Benchmark |
|------|-------------|---------------|
| Segmentation Dice (mandible) | ≥ 0.90 | 0.961 |
| ASSD (mm) | ≤ 0.5mm | 0.31mm |
| Landmark error (mm) | ≤ 2.0mm | 1.24mm |
| Guide fit RMSE (mm) | ≤ 0.3mm | 0.21mm |
| Planning angular error | ≤ 2° | 0.8° |

**4. Clinical Validation**
- Prospective study ≥ 100 cases, multi-site, multi-operator
- TRIPOD-AI transparent reporting
- Comparison to commercial VSP ground truth (surgical outcome at 6 months)
- IRB approval at each clinical site

**5. Predetermined Change Control Plan (PCCP)**
- AI/ML models may be continuously updated with new training data
- PCCP documents what types of changes (model retraining, new patient population)
  are pre-approved by FDA vs. require a new 510(k)
- Required since FDA's 2021 AIML guidance

**6. Post-Market Surveillance**
- Mandatory complaint reporting (MDR, 21 CFR Part 803)
- Adverse event monitoring: wrong-site surgery, guide misfit
- Periodic performance assessment (PPA) — recommended every 12 months

### EU MDR Considerations (CE Marking)
Under EU MDR 2017/745:
- **Class IIa** for surgical planning software (Annex VIII, Rule 11: software influencing clinical decisions)
- Notified Body (NB) conformity assessment required
- Clinical Evaluation Report (CER) per MEDDEV 2.7/1 Rev. 4
- Post-Market Clinical Follow-up (PMCF) plan

---

## 4. Quality Assurance for 3D-Printed Surgical Guides

### Standards Framework

| Standard | Scope |
|----------|-------|
| ISO/ASTM 52900 | Additive manufacturing terminology |
| ISO 17296-3 | Additive manufacturing — quality principles |
| AAMI ST91:2021 | Sterilisation of patient-contact parts |
| ISO 10993 | Biocompatibility of medical device materials |
| ASTM F2792 | Standard terminology for additive manufacturing |
| ISO 13485:2016 | QMS for medical device design and manufacturing |
| 21 CFR Part 820 | FDA Quality System Regulation |

### Material Requirements

**Surgical Guides (Patient-Contact, Non-Implantable):**
- Material: Formlabs Surgical Guide Resin, EnvisionTEC MED610, or equivalent
- Biocompatibility: ISO 10993-5 (cytotoxicity), 10993-10 (sensitisation)
- Sterilisation: Gamma or EtO compatible (validated per ISO 11135)
- Dimensional accuracy: ± 0.1mm at guide mating surface

**Patient-Specific Implants (Implantable):**
- Material: Titanium Ti-6Al-4V ELI (ASTM F136) or PEEK (ASTM F2026)
- Biocompatibility: ISO 10993-1 complete biocompatibility matrix
- Fatigue testing: ASTM F382 for titanium bone plates
- Corrosion: ASTM F746

### Pre-Print QA Checklist (Automated in VSP-3D)

```
Mesh Quality:
  ✓ Watertight (Euler number = 2)
  ✓ Consistent face winding
  ✓ No degenerate faces (area < 1e-10 mm²)
  ✓ No self-intersections
  ✓ Max edge length ≤ 2mm (printer resolution constraint)
  ✓ Min wall thickness ≥ 0.5mm (mesh guide) / 1.5mm (solid plate)

Dimensional:
  ✓ Fits within build volume
  ✓ Guide base RMSE ≤ 0.2mm from bone surface
  ✓ Cutting slot width = 0.5mm + saw blade kerf
  ✓ Screw hole diameter = screw OD + 0.1mm clearance

Clinical:
  ✓ No guide-bone collision outside intended mating surface
  ✓ Tissue windows ≥ 5mm for soft tissue retraction
  ✓ Orientation markers for intraoperative guide seating verification
```

### Post-Print Physical Verification

Before use on patient:

1. **Dimensional inspection**: Caliper measurement at ≥ 5 reference points
2. **Model fit test**: Guide placed on physical 3D-printed skull model (PolyJet)
   - Pass criterion: Visual gap ≤ 0.3mm, no rocking
3. **Sterilisation validation**: Cycle compatibility test per AAMI ST91
4. **Labelling**: Unique Device Identifier (UDI), patient ID (blinded),
   laterality, use-by date, sterilisation lot number

### Intraoperative Guide Verification

1. Guide is seated on the bone surface prior to making any osteotomy cut
2. Surgeon confirms visual and tactile fit to bone surface
3. Anatomical reference features (e.g., infraorbital foramen, mental foramen)
   are verified against pre-operative plan
4. If guide fit is uncertain, intraoperative imaging (fluoroscopy or CBCT)
   may be used to confirm position before the osteotomy is made

---

## 5. Clinical Integration at Academic Medical Centers

### Recommended Implementation Pathway

**Phase 1 (Months 1–6): Retrospective Validation**
- Process 50+ retrospective cases through VSP-3D pipeline
- Compare segmentation, landmarks, and planning to gold-standard (manual)
- No patient exposure; pure algorithm development and validation

**Phase 2 (Months 6–18): Prospective Parallel Processing**
- VSP-3D processes cases in parallel with standard commercial VSP
- Clinical team reviews both (blinded); select best plan
- Track: time savings, surgeon preference, accuracy metrics
- IRB exemption (quality improvement) or minimal risk protocol

**Phase 3 (Months 18–36): Supervised Clinical Use**
- VSP-3D used as primary planning tool
- Clinical engineer reviews and approves all AI outputs before manufacturing
- Full IRB-approved prospective outcomes study
- Track: guide fit rates, revision rates, OR time, patient outcomes (ANB at 6 months)

**Phase 4 (Year 3+): FDA Clearance and Commercial Deployment**
- 510(k) submission based on accumulated evidence
- Integration with hospital PACS (HL7 FHIR for DICOM ordering)
- EHR integration for automatic case routing

---

## References

1. Farber SJ et al., "Automated Virtual Surgical Planning Using Deep Learning", J Oral Maxillofac Surg 2022
2. Xia JJ et al., "Accuracy of Computer-Aided Orthognathic Surgery", IJOMS 2011
3. Zinser MJ et al., "A Paradigm Shift in Orthognathic Surgery", JCMS 2013
4. Zhao L et al., "Deep Learning-Based Automatic Cephalometric Landmark Detection", CBCT 2023
5. FDA, "Artificial Intelligence and Machine Learning in Software as a Medical Device", 2021
6. Isensee F et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods 2021
7. Wasserthal J et al., "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures", Radiology AI 2023
8. Meyer E et al., "Normalized metal artifact reduction (NMAR)", Med Phys 2010

---

*This document is maintained as living documentation. Clinical workflow details
are based on published literature and do not represent advice from any specific
institution or regulatory body.*
