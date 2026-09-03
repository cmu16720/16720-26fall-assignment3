# 1. Programming Assignment 3: Geometry in Motion

PA3 develops a complete planar-motion workflow: normalized DLT, robust homography estimation, a separate coarse-to-fine translation study, and a self-captured Geometry-in-Motion application. The assignment is worth 100 base points. Part E and optional showcase participation share one 10-point extra-credit cap.

## 2. Schedule and expected effort

- Release: October 27, 2026.
- Due: November 24, 2026.
- Target effort: 8–10 hours after setup.
- Required platform: Google Colab-compatible Python; the core checks run on CPU.

Use the Part C baseline capture as the Part D baseline to avoid duplicate recording. The three controlled variants may be short clips from the same capture session.

## 3. Setup

For Colab, download the complete course-supplied `pa3_student.zip`; the notebook
alone omits the package, tests, data setup, and requirements. Upload
`assignment3/notebooks/pa3_starter.ipynb` to a fresh runtime, run its first
environment cell, and choose the **complete, unopened `pa3_student.zip`** when
prompted. The cell validates ZIP paths, extracts beneath `/content`, changes to
the release root, and installs `requirements-colab.txt`. If Colab requests a
restart, restart and rerun that cell. Then enable the notebook's required B4
`wall` download before producing graded B4 evidence.

From the repository root, install the development dependencies, generate deterministic data, and fetch the checksum-verified Oxford `wall` sequence required by graded B4:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python assignment3/data/setup_data.py --synthetic --b4
```

The starter notebook contains the equivalent setup. Its network toggle may remain off for CPU smoke checks, but the synthetic fallback is not a substitute for the graded `wall` experiment. The legacy-compatible command below fetches all configured Oxford data: required `wall` plus optional/debug `graf`:

```bash
.venv/bin/python assignment3/data/setup_data.py --external
```

`wall` is used only for the required B4 planar-sequence evaluation. `graf` is optional debugging/evaluation data. Neither sequence satisfies the self-capture requirements in Parts C/D.

## 4. Files students edit

- `student/pa3.py`: the five required implementation regions marked `TODO(student)`.
- `notebooks/pa3_starter.ipynb`: experiments, visualizations, and artifact workflow.
- `answer_template.tex`: written responses and figure/table references.

Do not edit supplied helpers or public tests. Part C integration code may be added to the notebook or to additional clearly named source files.

## 5. Fixed numerical conventions

- Points are float64 arrays of shape `(N, 2)` in `(x, y)` order.
- Images are indexed `image[y, x]`; pixel centers have integer coordinates.
- `H` maps source points to destination points; warping inverse-samples.
- DLT uses the stated column-major vectorization and `reshape(..., order="F")`.
- Normalized points have centroid zero and mean radius `sqrt(2)`.
- Symmetric transfer error is the mean of squared forward/backward distances.
- The RANSAC threshold is in pixels and is squared before comparison.
- RANSAC uses an exact finite-population adaptive budget, seed `16720`, deterministic ties, and a 5000-trial cap.
- LK uses four levels unless an experiment says otherwise, at most 50 iterations per level, and tolerance `1e-3` level pixels.
- B4 is a separate translation study with `beta=0`; Part C uses the supplied SIFT-plus-homography baseline.

## 6. Running tests

Run the released public tests against your student file with:

```bash
python -m pytest assignment3/tests/test_pa3_public.py -q
```

Public tests cover contracts and representative cases. Additional staff tests cover edge cases. Passing public tests is necessary but not sufficient for full credit.

## 7. Capture, privacy, and fallback policy

Record a planar target without non-consenting people, private documents, personal screens, unsafe behavior, or prohibited locations. The default and expected path is the student's own capture.

The concrete course-authored package at `data/fallback/` may replace only personal recording, and only after written, advance staff approval for an access, privacy, safety, or accessibility barrier. Request approval before starting Part C and no later than seven calendar days before the deadline when the barrier is known. Follow `data/fallback/README.md`; a valid approval has identifier `PA3-FB-YYYY-NNN`. The request need name only the category and affected requirement—never a diagnosis or unnecessary personal details.

Approval does not waive implementation, predictions, controlled variants, manual corner annotation, metrics, comparisons, failure analysis, Hero Result, reflection, or acknowledgments. Without a valid approval identifier, the supplied clips do not satisfy Parts C/D. The package is release-included and can be regenerated and SHA-256 audited with:

```bash
python assignment3/data/setup_data.py --fallback
```

## 8. Submission summary

Submit:

- completed notebook and required source files;
- one LaTeX-generated answer PDF using stable A1–D3 labels;
- required plots, tables, manual-corner measurements, and diagnostic views;
- one 10–30 second Hero Result and a static preview;
- AI-use/comprehension reflection and complete acknowledgments.

The Hero Result may be H.264 MP4, GIF, or self-contained HTML, at most 30 seconds and 75 MB. Video must be at most 720p/30 fps; H.264 should use yuv420p. Retain raw captures locally and submit only the minimum requested evidence. Static preview formats are PNG/JPG and should be at least 1280 pixels on the longer side when source material permits.

## 9. Integrity boundary

Generative AI is prohibited for Part A and the required Part B implementations. It is permitted in Parts C–E as stated, with disclosure and comprehension. Instructor solutions and staff tests are never part of the student release.

## 10. Support and expected outputs

Consult `expected_outputs.md` for non-revealing numerical ranges and failure diagnostics. Ask course staff when a coordinate direction, capture accommodation, or artifact format remains unclear; do not change a required interface silently.
