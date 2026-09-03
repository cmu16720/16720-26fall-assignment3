# 1. PA3 expected-output guide

This guide provides debugging ranges without revealing required implementations. Small floating-point differences are normal; matching a printed number is not a substitute for satisfying the contract and explaining the result.

## 2. Normalization checks

For the A2 square, normalized centroid should be numerically zero, mean radius should be `sqrt(2)`, and transformed points should lie at coordinates with absolute value one. The returned last homogeneous coordinate should be one.

If all input points coincide, normalization should raise a useful `ValueError` rather than divide by zero.

## 3. DLT checks

- The exact non-symmetric four-point public case should have corner error below `1e-8 px`.
- A solution that accidentally transposes `H` fails by a large margin on that case even if identity/translation tests pass.
- With 100 inlier points and one-pixel measurement noise, corner RMSE around `0.4–2 px` is reasonable.
- Degenerate minimal sets should be rejected rather than produce a finite-looking arbitrary matrix.
- Multiplying `H` by any nonzero scalar must not change projected points or symmetric error.

## 4. RANSAC checks

Using the released generator, 3-pixel threshold, confidence 0.99, and RNG seed 16720:

| Outlier fraction | Typical RANSAC corner RMSE | Expected trend |
|---:|---:|---|
| 0.0 | below 2 px | few trials |
| 0.2 | below 3 px | modest budget |
| 0.4 | below 3 px | larger budget |
| 0.6 | below 4 px | hundreds of trials may be reasonable |

Precision should normally exceed 0.95 and recall should usually exceed 0.85. The exact trial count can change with a correct tie/degeneracy implementation, but identical seed and inputs must reproduce identical output within one implementation.

Common diagnostics:

- consistently huge error at every outlier rate: transform direction or reshape error;
- good direct DLT but bad RANSAC: threshold was not squared, wrong error units, or refit omitted;
- inconsistent repeated runs: global random state or nondeterministic ties;
- premature termination: IID budget used instead of the required finite-population probability.

## 5. Lucas–Kanade checks

- Full-resolution `(8,-5)` becomes `(1,-0.625)` at the coarsest level of a four-level pyramid and becomes `(2,-1.25)` at level two. The separate scalar check “8 pixels at level two” is exactly 2 pixels.
- A one-pixel positive-horizontal translation should produce a positive first update and finish near `(1,0)`.
- Four-level LK on the released `(8,-5)` pair should finish within `0.15 px` of truth from zero initialization.
- The one-level run may converge to a wrong local minimum on the same released texture; that contrast is intentional.
- A flat patch should report singularity or insufficient rank without crashing.
- An additive-brightness case should retain a larger photometric residual even if its endpoint estimate remains visually plausible.
- The graded real-sequence row uses Oxford `wall` image 1 to image 2, `H1to2p`, and ROI `x=68:132, y=268:332`; a clearly labeled synthetic fallback is smoke-only and earns no B4.REAL credit.
- With the staff reference, the four-level `wall` run has subpixel ROI-center endpoint error while the one-level zero-initialized run fails by more than 20 pixels. The nonzero four-corner residual is expected because a translation cannot exactly reproduce the local projective warp.

## 6. Geometry-in-Motion baseline checks

A valid diagnostic normally shows at least 10 inliers, inlier ratio at least 0.25, source coverage at least 0.15, median symmetric error at most 3 px, and a finite convex target quadrilateral. These gates are necessary diagnostics, not proof of correct geometry; compare manually annotated corners on selected frames.

The default C/D evidence uses self-captured footage. An approved fallback run must name only a valid `PA3-FB-YYYY-NNN` identifier and use all four release clips. Its manual annotations should be compared with, not replaced by, the package's oracle corners.

The stale state may display the last map for at most three frames and must be visibly labeled. After that, hide the overlay until a global reinitialization passes every gate.

## 7. Artifact audit

Before submission verify that the 10–30 second result remains readable without notebook context, includes synchronized diagnostics, and visibly states an assumption and non-claim. Check H.264/yuv420p, at most 720p/30 fps, and file size at most 75 MB.
