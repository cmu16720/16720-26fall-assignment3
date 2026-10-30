# 1. PA3 expected-output guide

This guide gives debugging ranges without giving away the required code. Small floating-point differences are normal. A matching number does not mean your code is correct, and you still need to explain the result.

## 2. Normalization checks

For the A2 square, the normalized centroid should be zero up to rounding, the mean radius should be `sqrt(2)`, and every transformed coordinate should have absolute value one. The last homogeneous coordinate should be one.

If all input points are the same, normalization should raise a useful `ValueError` instead of dividing by zero.

## 3. DLT checks

- The exact non-symmetric four-point public case should have corner error below `1e-8 px`.
- A solution that transposes `H` by mistake fails that case by a large margin, even if the identity and translation tests pass.
- With 100 inlier points and one pixel of measurement noise, a corner RMSE of about `0.4` to `2 px` is reasonable.
- Degenerate minimal sets should be rejected, not turned into a finite but arbitrary matrix.
- Multiplying `H` by any nonzero scalar must not change the projected points or the symmetric error.

## 4. RANSAC checks

With the released generator, a 3-pixel threshold, confidence 0.99, and RNG seed 16720:

| Outlier fraction | Typical RANSAC corner RMSE | Expected trend |
|---:|---:|---|
| 0.0 | below 2 px | few trials |
| 0.2 | below 3 px | modest budget |
| 0.4 | below 3 px | larger budget |
| 0.6 | below 4 px | hundreds of trials can be reasonable |

Precision should normally be above 0.95 and recall usually above 0.85. The exact trial count can differ between correct implementations that handle ties and degenerate samples differently, but the same seed and inputs must give the same output every time for one implementation.

`ransac_homography` returns a `numpy.bool_` mask of shape `(N,)`. Its `info` dictionary must contain `trials`, `adaptive_budget`, `best_sample`, and `errors`. Return `best_sample` as a tuple of four `int`. If it is an array, the public determinism check raises "truth value of an array is ambiguous" instead of comparing two runs. If the budget runs out and no four-point model has a consensus of at least four correspondences, raise `RuntimeError`. The supplied Part C cell catches it and marks the frame invalid.

Common symptoms:

- Large error at every outlier rate: wrong transform direction or reshape.
- Direct DLT is fine but RANSAC is not: the threshold was not squared, the error units are wrong, or the final refit was skipped.
- Repeated runs disagree: global random state or tie-breaking that changes between runs.
- RANSAC stops too early: the IID budget was used instead of the required finite-population probability.

## 5. Lucas-Kanade checks

`lucas_kanade_refine` returns `(p, history)`. The public test and the supplied notebook cells read `history` by key, so the record format is part of the contract:

| Key | Present on | Value |
|---|---|---|
| `phase` | every record | `"level_start"`, `"iteration"`, or `"level_end"` |
| `level` | every record | pyramid level, with zero at full resolution |
| `p_level` | every record | estimate in that level's pixels |
| `p_full` | every record | `p_level * 2 ** level` |
| `status` | every record | `"start"`, `"iterate"`, `"converged"`, `"max_iterations"`, `"singular"`, `"ill_conditioned"`, or `"insufficient_overlap"` |
| `photometric_rmse` | iteration records | RMS residual over the pixels valid in both images for that iteration |

The list starts with the `level_start` record of the coarsest level and ends with the `level_end` record of level zero. For four levels, `history[0]["level"] == 3` and `history[-1]["level"] == 0`. Extra keys are fine. A missing key fails the public test even when the returned translation is correct.

- A full-resolution shift of `(8,-5)` is `(1,-0.625)` at the coarsest level of a four-level pyramid and `(2,-1.25)` at level two. The separate scalar check, 8 pixels at level two, is exactly 2 pixels.
- A one-pixel shift in the positive x direction should give a positive first update and finish near `(1,0)`.
- Four levels on the released `(8,-5)` pair should finish within `0.15 px` of the truth, starting from zero.
- One level can converge to a wrong local minimum on the same texture. That contrast is intended.
- A flat patch should end with status `singular` without crashing, instead of inverting a rank-deficient normal matrix. Fewer than `LK_MIN_VALID` pixels valid in both images gives `insufficient_overlap`, and a design-matrix condition number above `LK_MAX_CONDITION` gives `ill_conditioned`.
- An additive brightness change should leave a larger photometric residual, even if the endpoint looks plausible.
- The graded sequence row uses Oxford `wall` image 1 to image 2, `H1to2p`, and ROI `x=68:132, y=268:332`. The labeled synthetic smoke check earns no B4.REAL credit.
- With the staff reference, the four-level `wall` run has subpixel endpoint error at the ROI center, while the one-level run from zero is off by more than 20 pixels. Some corner error remains because a translation cannot match the local perspective warp exactly.

## 6. Geometry-in-Motion baseline checks

A valid frame normally shows at least 10 inliers, an inlier ratio of at least 0.25, inlier coverage of at least 15% of the target, a square root of the median symmetric transfer error of at most 3 px (a median error of at most 9 px squared), and a finite convex target quadrilateral. Passing these checks does not prove the geometry is correct. Compare against hand-marked corners on selected frames.

Parts C and D use your own footage by default. An approved fallback run gives only a valid `PA3-FB-YYYY-NNN` ID and uses all four supplied clips. Compare your hand-marked corners with the package's known corners. Do not use the known corners in place of your own marking.

The `STALE` state may show the last result for at most three frames and must be labeled. After that, hide the overlay until a whole-frame search passes every check.

## 7. Hero Result check

Before submitting, check that the 10 to 30 second result makes sense without the notebook, includes synchronized diagnostics, and shows a short statement of your assumptions and the method's limitations. Check H.264/yuv420p, at most 720p and 30 fps, and a file size of at most 75 MB.
