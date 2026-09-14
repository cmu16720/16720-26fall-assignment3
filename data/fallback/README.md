# 1. PA3 Approved Accessibility Fallback

## 2. Who may use this package

This package replaces only recording your own footage for Parts C and D. You may use it only after course staff give written approval in advance for an access, privacy, safety, or accessibility barrier. Lack of time, preferring not to use your own equipment, or a failed recording does not count as approval.

## 3. How to get approval

1. Ask before starting Part C, and no later than seven days before the PA3 deadline if you already know about the barrier.
2. Send a private message to the instructor and the PA3 lead through the course's private support channel, with the subject `PA3 fallback request: <Andrew ID>`.
3. Name only the category of barrier (access, privacy, safety, or accessibility) and the requirement it affects. Do not send a diagnosis or other personal details.
4. Wait for written approval with an ID of the form `PA3-FB-YYYY-NNN`. A public post, no reply, or a suggestion from a classmate or TA does not count.
5. In the notebook, set `USE_APPROVED_FALLBACK=True` and enter the ID in `FALLBACK_APPROVAL_ID`. In the acknowledgments of your PDF, list only the ID, the approval date, and the approving staff member's initials.

If the private support channel is not available, email the instructor and the PA3 lead from the address registered for the course, with the same subject and details.

## 4. What approval covers

Approval only replaces your own recording. Everything else in Parts C and D is still required, including predictions, hand-marked corners on 5 to 10 frames per condition, both method comparisons, the Hero Result, and the reflection.

Use `baseline.mp4` for Part C. Use all four clips for Part D: `baseline.mp4`, `fast_motion.mp4`, `occlusion.mp4`, and `illumination.mp4`. All clips have the same duration, frame rate, resolution, target, and background. Each variant changes only the factor in its filename.

## 5. Hand-marked corners and known corners

Mark the corners by hand on the chosen frames before you open `ground_truth.json`. The file gives the exact homographies and ordered corners used to make the clips, so you can check your work. It does not replace your hand marking. Report how far your marked corners are from the known corners, and name the source of the corners you use in later tables.

The nine suggested frames are listed in `evaluation_frames`. Corners are ordered top-left, top-right, bottom-right, bottom-left, in `(x,y)` coordinates.

## 6. Files

- `planar_target.png`: the target image.
- `baseline.mp4`: slow planar motion.
- `fast_motion.mp4`: only the motion between frames changes.
- `occlusion.mp4`: only the occlusion of the target changes.
- `illumination.mp4`: only the global brightness gain and offset change.
- `ground_truth.json`: homographies, corners, and changed-factor values for every frame.
- `manifest.json`: file sizes and SHA-256 hashes for every file in the package.
- `LICENSE.md`: license and source of these course-made files.

To regenerate the package and check its hashes, run `setup_data.py --fallback` from the `data` folder's parent directory, for example `python data/setup_data.py --fallback`.

Do not edit the clips. Save everything you create in the `artifacts/` folder.
