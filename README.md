# 16-720 Computer Vision: Assignment 3

**Geometry in Motion**

| | |
|---|---|
| Release | October 27, 2026 |
| Due | November 24, 2026, 11:59 pm ET |
| Credit | 100 points, plus up to 10 optional points |
| Expected effort | 10+ hours |
| Compute | Google Colab, CPU only |
| Final artifact | 10 to 30 second Hero Result and a still image, at most 75 MB |

Build a planar-motion pipeline with normalized DLT, finite-population RANSAC,
and coarse-to-fine Lucas-Kanade. Then use it to track a planar target in video
you record yourself, and measure where it works and where it fails.

## Getting started

The [assignment instructions (PDF)](assignment_instructions.pdf) have every
question, requirement, and grading rule. This README covers setup and submission.
The starter notebook holds the code you run for each part. Read the full handout
before starting.

- Work through the [starter notebook](pa3_starter.ipynb).
- Implement the required functions in [`student/pa3.py`](student/pa3.py).
- Write your answers in [`answer_template.tex`](answer_template.tex).
- Run [`tests/test_pa3_public.py`](tests/test_pa3_public.py) for public checks.
- See [`data/PROVENANCE.md`](data/PROVENANCE.md) for data sources and privacy notes.

## Local setup

Python 3.10 or newer is required.

```bash
git clone https://github.com/cmu16720/16720-26fall-assignment3.git
cd 16720-26fall-assignment3
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python data/setup_data.py --synthetic --b4
# Run after implementing B2 to B4. The untouched starter fails these tests.
.venv/bin/python -m pytest tests/test_pa3_public.py -q
```

If `uv` is not installed, create a standard virtual environment and install the
project with `python -m pip install -e '.[dev]'`.

`--synthetic` needs no network. `--b4` downloads the checksum-verified Oxford
`wall` sequence needed for the graded B4 row.

## Collaboration, academic integrity, and generative AI

Talk to your classmates. Working through derivations, debugging strategies, and
experiment design together is one of the best ways to learn this material, and
we encourage it, as we do office hours. Share ideas, not work. Everything you
submit must be your own, you may not read or copy another student's solutions,
and you must list everyone you worked with in your acknowledgments.

| Work | Policy |
|---|---|
| Part A | Generative AI may not generate, explain, solve, check, or rewrite answers. |
| Part B required functions | Generative AI may not generate, complete, translate, explain, debug, or rewrite the required functions. |
| Parts C, D, and E | Generative AI is allowed and encouraged, with disclosure. You are responsible for every part of your system. |

Do not call an OpenCV homography estimator inside a required Part B function.
In the written PDF, list your collaborators, outside resources, datasets,
pretrained weights, and every allowed use of generative AI. Section 4 of the
[assignment PDF](assignment_instructions.pdf) has the full policy.

## Capture and privacy

Parts C and D need footage you record yourself. Do not record people who have
not agreed to it, private documents, personal information on screens, or places
where recording is unsafe or not allowed. The clips in `data/fallback/` are for
debugging. They can replace your own recording only if course staff give written
approval in advance for an access, privacy, safety, or accessibility barrier. See
[`data/fallback/README.md`](data/fallback/README.md) and Section 10 of the
[assignment PDF](assignment_instructions.pdf).

## Submission

Submit on Gradescope by November 24, 2026, 11:59 pm ET.

Upload to **PA3 Written**:

1. one PDF compiled from `answer_template.tex`, with your acknowledgments and AI disclosure.

Upload to **PA3 Code**:

1. `pa3.py` with your Part B functions, and any other source files you wrote,
2. the executed notebook with its outputs,
3. your Hero Result (H.264/yuv420p MP4, GIF, or self-contained HTML), 10 to 30
   seconds long and at most 75 MB,
4. one still image from the Hero Result.

**Optional Slack showcase (+1).** After you submit on Gradescope, post your Hero
Result file, or a link to your HTML page, in the course Slack channel
`#showcase-hw3`, within 24 hours after the deadline. The Slack post does not
replace the Gradescope upload.

Before submitting, run the public tests in a clean process, run the notebook in
a fresh runtime, check the compiled PDF, and play the Hero Result outside Colab.
Section 12 of the [assignment PDF](assignment_instructions.pdf) has a checklist
of what each part needs.

## Colab editing and PDF figures

Upload `pa3_starter.ipynb` to Colab, then upload the repository ZIP downloaded
from GitHub when the first cell asks for it. In the Files pane, open
`student/pa3.py`, save your edits, and run the notebook's reload cell before
rerunning the cells that use your code. Download your edited code often, because
saving the notebook does not back up separate runtime files. The backup cell at
the end downloads your code and figures. Download the executed notebook separately
with File → Download → Download .ipynb.

The notebook saves supplied figures to `artifacts/`. Save your own figures there
too, upload them next to `answer_template.tex` in Overleaf, and replace each
evidence box with `\includegraphics[width=\linewidth]{your_figure.png}`. Compile
with pdfLaTeX and check every answer page.
