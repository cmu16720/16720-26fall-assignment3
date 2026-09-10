# 16-720 Computer Vision — Assignment 3

**Geometry in Motion**

| | |
|---|---|
| Release | October 27, 2026 |
| Due | November 24, 2026, 11:59 pm ET |
| Credit | 100 points, plus up to 10 optional points |
| Expected effort | 8–10 hours |
| Compute | Google Colab; CPU is sufficient |
| Final artifact | 10–30 second Hero Result plus a static preview, at most 75 MB |

Build a robust planar-motion pipeline—normalized DLT, finite-population RANSAC,
and coarse-to-fine Lucas–Kanade—then use it to create a tracked planar
application from video you shoot yourself, and measure where it breaks.

## Getting started

Read the [assignment instructions (PDF)](assignment_instructions.pdf) before editing code.
It is the authoritative source for all requirements, numerical contracts, grading,
and policy. The notebook is its executable companion.

- Work through the [starter notebook](pa3_starter.ipynb).
- Implement the required functions in [`student/pa3.py`](student/pa3.py).
- Write responses in [`answer_template.tex`](answer_template.tex).
- Use [`tests/test_pa3_public.py`](tests/test_pa3_public.py) for public checks.
- See [`data/PROVENANCE.md`](data/PROVENANCE.md) for data sources and privacy notes.

## Local setup

Python 3.10 or newer is required.

```bash
git clone https://github.com/cmu16720/16720-26fall-assignment3.git
cd 16720-26fall-assignment3
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python data/setup_data.py --synthetic --b4
# Run after implementing B2–B4; the untouched starter is expected to fail.
.venv/bin/python -m pytest tests/test_pa3_public.py -q
```

If `uv` is unavailable, create a standard virtual environment and install the
project with `python -m pip install -e '.[dev]'`.

`--synthetic` needs no network. `--b4` fetches the checksum-verified Oxford
`wall` sequence required by the graded B4 evaluation.

## Collaboration, academic integrity, and generative AI

Talk to your classmates: discussing derivations, debugging strategies, and
experimental design together is one of the best ways to learn this material, and
we encourage it, as we do office hours. Share ideas, not artifacts — everything
you submit must be produced by you, you may not read or copy another student's
solutions, and you must list everyone you worked with in your acknowledgments.

| Work | Policy |
|---|---|
| Part A | Generative AI may not generate, explain, solve, check, or rewrite responses. |
| Part B required implementations | Generative AI may not generate, complete, translate, explain, debug, or rewrite the required functions. |
| Parts C, D, and E | Generative AI is permitted and encouraged with disclosure; you remain responsible for every component. |

Do not call an OpenCV homography estimator inside a required Part B function.
Disclose collaborators, outside resources, datasets, pretrained weights, and
every permitted use of generative AI in the written PDF. See the
[assignment PDF, Section 4](assignment_instructions.pdf)
for the complete policy.

## Capture and privacy

Parts C and D require footage you shoot yourself. Do not record non-consenting
people, private documents, personal information on screens, or unsafe or
prohibited locations. Supplied clips under `data/fallback/` are debugging
material; they replace personal recording only with written, advance staff
approval for an access, privacy, safety, or accessibility barrier. See
[`data/fallback/README.md`](data/fallback/README.md) and the
[assignment PDF, Section 10](assignment_instructions.pdf).

## Submission

Submit through the course submission system:

1. `pa3.py` with your Part B implementations;
2. the completed notebook with required outputs retained;
3. one PDF compiled from `answer_template.tex`;
4. one Hero Result (H.264/yuv420p MP4, GIF, or self-contained HTML), 10–30
   seconds and at most 75 MB, plus one static preview;
5. all required acknowledgments and disclosures.

Before submitting, run the public tests in a clean process, execute the notebook
from a fresh runtime, inspect the compiled PDF, and play the exported artifact
outside Colab. Verify every detailed requirement against the
[assignment PDF, Section 12](assignment_instructions.pdf).

## Colab editing and PDF evidence

Upload `pa3_starter.ipynb` to Colab, then upload the complete repository ZIP when
the first cell prompts for it. In the Files pane, open `student/pa3.py`, save your
edits, and run the notebook's reload cell before rerunning dependent cells.
Download edited code regularly; saving the notebook does not back up separate
runtime files. Download the executed notebook separately with
File → Download → Download .ipynb.

The notebook saves supplied figures to `artifacts/`. Save your added figures there
too, upload them alongside `answer_template.tex` in Overleaf, and replace each
relevant evidence box with `\includegraphics[width=\linewidth]{your_figure.png}`.
Compile with pdfLaTeX and inspect every answer page.
