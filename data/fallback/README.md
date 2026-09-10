# 1. PA3 Approved Accessibility Fallback

## 2. Who may use this package

This package replaces only the act of recording Parts C/D footage. It may be used only after written, advance approval from course staff for an access, privacy, safety, or accessibility barrier. Lack of time, ordinary equipment preference, or a failed personal recording is not automatic approval.

## 3. Exact approval procedure

1. Request approval before starting Part C and no later than seven calendar days before the PA3 deadline when the barrier is known.
2. Send a private message to the instructor and PA3 lead through the course's announced private support channel with subject `PA3 fallback request — <Andrew ID>`.
3. Name only the applicable category—access, privacy, safety, or accessibility—and the assignment requirement affected. Do not send diagnoses or unnecessary personal details.
4. Wait for written approval containing an identifier of the form `PA3-FB-YYYY-NNN`. Silence, a public post, or a peer/TA suggestion is not approval.
5. In the notebook, set `USE_APPROVED_FALLBACK=True` and enter that identifier in `FALLBACK_APPROVAL_ID`. In the answer PDF acknowledgments, record only the identifier, approval date, and approving staff initials.

If the private support channel is unavailable, email the instructor and PA3 lead from the address registered for the course using the same subject and fields.

## 4. Scope of the waiver

Approval waives only personal media capture. It does not waive implementation, predictions made before evaluation, the baseline plus three one-factor controls, manual ordered-corner annotations on 5–10 frames per condition, external metrics, valid-frame metrics, direct-DLT/RANSAC and one-/four-level comparisons, failure analysis, Hero Result, reflection, or acknowledgments.

Use `baseline.mp4` for Part C. Use all four clips for Part D: `baseline.mp4`, `fast_motion.mp4`, `occlusion.mp4`, and `illumination.mp4`. The clips have identical duration, frame rate, resolution, target, and background. Each variant changes only the factor named by its filename.

## 5. Manual annotation and oracle metadata

Manually annotate the prescribed frames before using `ground_truth.json`. The JSON supplies exact course-authored homographies and ordered corners for verification; it does not replace the manual-annotation deliverable. Report the disagreement between your manual labels and the oracle, then use a clearly named metric source in later tables.

The recommended nine frames are listed in `evaluation_frames`. The corner order is top-left, top-right, bottom-right, bottom-left in `(x,y)` coordinates.

## 6. Asset inventory

- `planar_target.png`: matching target/reference image.
- `baseline.mp4`: nominal slow planar motion.
- `fast_motion.mp4`: only inter-frame displacement changes.
- `occlusion.mp4`: only target occlusion changes.
- `illumination.mp4`: only global brightness gain/bias changes.
- `ground_truth.json`: per-frame homographies, corners, and controlled-factor values.
- `manifest.json`: byte sizes and SHA-256 hashes for every payload file.
- `LICENSE.md`: course-authored asset license and provenance statement.

Regenerate and audit the package from the repository root with:

```bash
python assignment3/data/setup_data.py --fallback
```

Do not edit the source clips. Save all student-created outputs beneath `assignment3/artifacts/`.
