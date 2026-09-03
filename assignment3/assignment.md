# 1. Programming Assignment 3: Geometry in Motion

**Released:** October 27, 2026  
**Due:** November 24, 2026  
**Points:** 100 base + up to 10 extra-credit points  
**Target:** 8–10 hours  
**Platform:** Google Colab  
**Submit:** code, one LaTeX-generated PDF, diagnostics, and a Hero Result

## 2. Overview

Images flatten the world. Geometry lets us reason backward: which measurements could come from the same scene, which motion explains two views, and when a transformation is trustworthy. In this assignment you will derive homography and differential-motion constraints, implement robust planar estimation, study its failure boundaries, and create a tracked planar application.

Your final artifact must do more than show a plausible overlay. It must expose estimated motion, contain an informative failure or recovery, and state both the assumptions being made and what the result does not establish about the three-dimensional scene.

## 3. Learning objectives

After completing PA3, you should be able to:

1. derive DLT constraints under an explicit vectorization convention;
2. normalize coordinates and solve homogeneous least squares with SVD;
3. implement and diagnose finite-population RANSAC;
4. derive and solve the Lucas–Kanade brightness-constancy system;
5. distinguish geometric, photometric, and downstream visual error;
6. explain homography and differential-alignment assumptions; and
7. communicate geometric success and failure in a polished result.

## 4. Global conventions

- Cartesian point arrays have shape `(N, 2)`, order `(x, y)`, and dtype float64.
- Images are `(H, W)` or `(H, W, C)` and indexed `image[y, x]`.
- Pixel centers lie at integer coordinates.
- A reported homography maps source to destination: `x_dst ~ H @ x_src`.
- Warping uses inverse sampling; interpolation and validity are separate operations.
- Invalid homogeneous scales become invalid measurements, not clipped points.
- All required code must run in a fresh Colab CPU runtime.
- Shared random seed: `16720`.

## 5. Part A — THINK: Constraints, Estimation, and Motion — 25 points

Generative AI is **not permitted** to generate, explain, solve, check, or rewrite Part A. Use course materials and other resources allowed by the collaboration policy. Typeset all answers in LaTeX.

### 5.1 A1. Derive the DLT constraints — 9 points

Let corresponding points be

\[
x=[x,y,1]^T,\qquad x'=[x',y',1]^T,
\]

with `x' ~ Hx`, where

\[
H=\begin{bmatrix}h_{11}&h_{12}&h_{13}\\h_{21}&h_{22}&h_{23}\\h_{31}&h_{32}&h_{33}\end{bmatrix}.
\]

This assignment uses column-major vectorization:

\[
h=\operatorname{vec}_c(H)=[h_{11},h_{21},h_{31},h_{12},h_{22},h_{32},h_{13},h_{23},h_{33}]^T.
\]

Do not transpose `H`; only the vectorization convention differs from lecture.

1. Starting from homogeneous equality up to scale, eliminate the scale and derive two independent scalar equations contributed by one correspondence.
2. Collect coefficients in the stated order and write the resulting `2 x 9` matrix `A_i` satisfying `A_i h = 0`.
3. Let `h_r=vec_r(H)` be the row-major lecture vector and define `h_r=Ph`. Without listing 81 entries, state how the nonzero positions of `P` are determined and derive the relationship between `A`, `A_r`, and `P`.
4. State the minimum number of nondegenerate correspondences, explain why a homography has eight rather than nine degrees of freedom, and explain why collinear configurations are degenerate.
5. Formulate noisy DLT as total least squares, explain the smallest right-singular-vector solution, and state exactly how NumPy must reshape that vector without transposing the intended map.

### 5.2 A2. Normalize a small correspondence set — 5 points

Consider `(0,0)`, `(2,0)`, `(2,2)`, and `(0,2)`.

1. Compute the centroid and mean Euclidean distance from it.
2. Construct the similarity transform `T` that centers the points and makes their mean distance `sqrt(2)`.
3. Transform all four points explicitly.
4. If normalized DLT estimates `H_tilde` from `x_tilde=Tx` and `x'_tilde=T'x'`, derive the conversion back to image coordinates.
5. Explain why normalization improves conditioning and why it does not make least-squares DLT robust to mismatches.

### 5.3 A3. RANSAC sampling without replacement — 5 points

Each trial samples `s` distinct correspondences uniformly without replacement from exactly `M` correspondences, of which exactly `I` are inliers.

1. Give the exact all-inlier probability using binomial coefficients or a finite-population product.
2. Using
   \[
   N=\left\lceil\frac{\log(1-p)}{\log(1-q)}\right\rceil,
   \]
   compute `N` for `M=40`, `I=20`, `s=4`, and `p=0.99`. Compare with `q_iid=(I/M)^s` and explain the direction of the difference.
3. If the current best consensus contains 26 of 40 correspondences, should an adaptive budget increase or decrease? Explain why optimistic consensus can stop too early.
4. Give one case where a high-consensus homography is the wrong model for the intended target.

### 5.4 A4. Local motion with additive brightness change — 6 points

Suppose a patch undergoes translation `(u,v)` and constant additive brightness change `beta`:

\[
I(x+u,y+v,t+1)=I(x,y,t)+\beta.
\]

1. Use a first-order Taylor approximation to derive `I_x u + I_y v - beta = -I_t`. Recover ordinary brightness constancy as a special case.
2. Explain why one pixel cannot determine all three unknowns and state the spatial gradient variation required for a well-constrained patch system.
3. For gradient rows `(1,0)`, `(0,1)`, `(1,1)` and temporal derivatives `-1,-2,-4`, construct `A theta=b` for `theta=[u,v,beta]^T` and solve exactly.
4. Explain what the eigenvalues and conditioning of `A^T A` reveal about ambiguity.
5. Predict whether single-scale LK or descriptor matching with RANSAC fails first as displacement increases. Revisit this prediction in D1.

## 6. Part B — BUILD: Robust Planar Motion — 35 points

Generative AI is **not permitted for the required implementations**. You may use NumPy, the explicitly supplied helpers, OpenCV documentation, SciPy documentation, and course materials. Do not call an OpenCV homography estimator inside a required function.

### 6.1 B1. Read the geometry pipeline — 5 points

Inspect the numbered cells in `pa3_starter.ipynb` and answer:

1. **B1.1:** In the coordinate-convention cell, what shapes do `xy` and `to_homogeneous(xy)` have, which columns store `x` and `y`, and why does point projection transpose the row array around matrix multiplication?
2. **B1.2:** The supplied warp accepts `H_src_to_dst`. Which direction does that matrix report, which direction is solved for each destination pixel, and why is this not a contradiction?
3. **B1.3:** What are the two outputs of `bilinear_sample`, and why must LK combine validity masks rather than treating interpolation as a validity test?
4. **B1.4:** Identify the two rows in the DLT scaffold that correspond to your A1 equations. Why must the recovered vector use `reshape((3,3), order="F")`?
5. **B1.5:** The pyramid list stores full resolution at index zero. Trace a full-resolution displacement of eight pixels to level two and back.

### 6.2 B2. Normalized DLT — 10 points

Implement:

- `normalize_points(xy) -> (normalized_xy_h, T)`;
- `estimate_homography_dlt(src_xy, dst_xy) -> H`; and
- `symmetric_transfer_error(H, src_xy, dst_xy) -> errors`.

The error for correspondence `i` is

\[
e_i=\tfrac12\left(\|\pi(Hx_i)-x'_i\|_2^2+\|\pi(H^{-1}x'_i)-x_i\|_2^2\right).
\]

It has units pixel squared. Invalid projection scales or a noninvertible map produce `inf`. Scale `H` so `H[2,2]=1` when safe; otherwise use unit Frobenius norm with deterministic sign. Reject design rank below eight. The tests include exact minimal data, noisy overdetermined data, permutation, inverse direction, scale invariance, input mutation, accidental transpose, and degeneracy.

### 6.3 B3. RANSAC homography estimation — 9 points

Implement:

```python
ransac_homography(src_xy, dst_xy, threshold, confidence, rng)
```

Return `(H, inlier_mask, info)`. `threshold` is in pixels; compare the B2 error with `threshold**2`. Each trial must:

1. sample four distinct correspondences without replacement;
2. reject repeated points and any collinear triplet in either minimal set;
3. score all correspondences with symmetric transfer error;
4. update the exact finite-population budget with `C(k,4)/C(N,4)`;
5. stop at that budget or 5000 trials, whichever occurs first;
6. break ties by larger consensus, lower median inlier error, lower total inlier error, then earlier trial; and
7. refit the final `H` from every returned inlier.

Use only the passed `numpy.random.Generator`. `info` must contain `trials`, `adaptive_budget`, `best_sample`, and final `errors`.

Evaluate direct DLT and RANSAC on supplied 100-point, 640×480 synthetic scenes with noise `sigma=1 px`, outlier fractions `0,0.2,0.4,0.6`, and seeds `16720` through `16724`. Use a 3-pixel threshold and confidence `0.99`. Report median corner-transfer RMSE, inlier precision/recall, and runtime. Interpret correctness and failure rather than optimizing a stochastic leaderboard.

### 6.4 B4. Coarse-to-fine Lucas–Kanade translation — 11 points

Implement:

```python
lucas_kanade_refine(template, image, p0, levels)
```

Return `(p, history)`. Level zero is full resolution and each next level halves both dimensions. Convert the initial estimate to the coarsest coordinates, then for levels coarse to fine:

1. sample `I_l(x+p_l)` and the supplied image gradients at the same coordinates;
2. retain only jointly valid pixels;
3. form `r(x)=T_l(x)-I_l(x+p_l)` and `A(x,:)=[I_x,I_y]`;
4. solve `A delta_p ~= r` with least squares, never an explicit normal-matrix inverse;
5. update `p_l <- p_l+delta_p`;
6. use at most 50 iterations and stop at update norm below `1e-3` level pixels;
7. report singular/ill-conditioned or insufficient-overlap termination in history; and
8. double the estimate when moving to the next finer level.

Use four levels for the required comparison. Verify 8 pixels becoming 2 at level 2, the one-pixel update sign, and estimates before/after every level. Compare one-level and four-level results on deterministic translations and the checksum-verified Oxford VGG `wall` sequence. Download the required asset with `python assignment3/data/setup_data.py --b4`; a synthetic fallback is a smoke check only and receives no B4 real-sequence credit.

For the graded sequence row, use `wall/img1.ppm` to `wall/img2.ppm`, `wall/H1to2p`, `p0=(0,0)`, and the half-open 64 × 64 ROI `x=68:132, y=268:332` in both frames. Define the reference endpoint as the ROI-center displacement under `H1to2p`. Report endpoint error and corner RMSE between the four translated ROI corners and their homography-warped locations. Plot the fixed ROI/patch, one- versus four-level photometric error by iteration, reference and predicted quads, and both geometric errors.

B4 is deliberately a **separate translation study**. It assumes `beta=0`; the A4 brightness parameter is not a hidden implementation requirement. Part C uses a feature-homography baseline. Clearly distinguish initialization failure from violation of brightness constancy.

## 7. Part C — CREATE: Geometry-in-Motion Experience — 20 points

Generative AI is permitted and encouraged for coding, debugging, visualization, and integration in Part C. You remain responsible for understanding and testing every submitted component.

### 7.1 C1. Choose a planar target and application — 4 points

Capture a 5–15 second video with a clearly visible poster, book cover, sign, painting, tabletop card, or facade. Downloaded or supplied sequences do not satisfy Parts C/D unless they are the course-authored `data/fallback/` package used under written, advance approval for an access, privacy, safety, or accessibility barrier.

Request fallback approval before starting Part C and no later than seven calendar days before the deadline when the barrier is known. Follow `data/fallback/README.md`: privately contact the instructor and PA3 lead, name only the applicable category and affected requirement, and wait for an approval identifier `PA3-FB-YYYY-NNN`. Set the notebook's approval guard and disclose only the identifier, date, and staff initials. Approval waives personal recording only; every implementation, prediction, four-condition study, manual annotation, metric, comparison, analysis, Hero Result, reflection, and acknowledgment remains required.

Choose AR overlay, plane stabilization, or another staff-approved planar-motion application. Document target, camera, setup/location, motion, lighting, frame count/resolution, and why a homography is plausible. Include one raw still and predict one model violation. Do not record non-consenting people, private documents, personal screens, unsafe activity, or prohibited locations.

### 7.2 C2. Build and improve the system — 10 points

Build the temporal baseline from the supplied per-frame components: up to 2000 SIFT features, L2 two-neighbor matching, ratio 0.75, mutual consistency, normalized-DLT RANSAC, and a 3-pixel threshold at a processing resolution no larger than 720 pixels on the long edge. Your baseline must estimate a target-to-frame homography per frame and implement the last-valid temporal state; the notebook does not supply the state machine.

A frame is valid only when its map is finite/invertible, has at least 10 inliers, inlier ratio at least 0.25, target-source inlier coverage at least 0.15, median symmetric error at most 3 pixels, and a finite convex projected quadrilateral. A stale result may be held for at most three frames with a visible `STALE` label; then hide it and attempt ungated global reinitialization. Show a scalar confidence based on inlier ratio, error, and spatial coverage.

The final pipeline must include initialization/reinitialization, temporal state, invalid-frame handling, and confidence. Add one substantive improvement beyond the baseline, such as automatic recovery, defensible smoothing, spatially distributed inlier logic, local photometric refinement, or an approved classical/learned comparison.

Produce a clean application view and a synchronized diagnostic view with correspondences/inliers, projected target boundary, confidence/error trace, and current validity state.

### 7.3 C3. AI-comprehension reflection — 6 points

In 200–300 words answer:

1. What did generative AI help implement or communicate?
2. How does the most important AI-assisted portion work, including inputs, outputs, and assumptions?
3. What did you change, debug, test, or redesign from the initially generated version, and why?
4. What check would detect a plausible-looking but geometrically incorrect result?

If you used no generative AI, state that clearly and use questions 2–3 to analyze the most important independently written or permitted third-party component and your design iterations. No-AI work receives equal credit.

## 8. Part D — UNDERSTAND & SHOW: Evidence, Failure, and Communication — 20 points

Generative AI is permitted and encouraged for visualization, with disclosure.

### 8.1 D1. Controlled robustness study — 7 points

Use your Part C clip as baseline, then record at least three short variants in the same session while changing one factor at a time: viewpoint, blur, illumination, occlusion, repeated texture, parallax/nonplanarity, reduced texture, or displacement. A student with documented fallback approval instead uses `baseline.mp4`, `fast_motion.mp4`, `occlusion.mp4`, and `illumination.mp4` from the package and performs the identical remaining workload. For each condition:

1. state what was held fixed and changed precisely enough to repeat the capture;
2. state a prediction before running it;
3. manually annotate the target’s ordered four corners on 5–10 representative frames;
4. report mean corner error or quadrilateral IoU plus a tracking metric such as valid-frame rate;
5. show a representative diagnostic; and
6. explain the outcome using sampling, conditioning, consensus, brightness constancy, aperture ambiguity, or model mismatch.

Include a compact table with target, motion, distance/viewpoint, lighting, controlled variable, duration, frame count, and resolution. At least one condition must compare direct DLT with RANSAC and at least one must compare one-level with four-level LK. Revisit A4. Inlier residual alone is self-consistency, not external geometric ground truth.

### 8.2 D2. When a good-looking warp is wrong — 5 points

Choose one success and one failure. For each, show the input frame, inliers, projected target boundary, application output, and synchronized error/confidence trace. Explain:

1. whether the dominant cause is measurement noise, outliers, optimization, interpolation, or invalid scene model;
2. why visual plausibility is insufficient evidence of geometric correctness; and
3. what additional calibrated-camera, epipolar, stereo, depth, or multi-view information would support a genuinely 3D conclusion.

### 8.3 D3. Hero Result — 8 points

Create a polished 10–30 second H.264 MP4, GIF, or self-contained HTML artifact containing:

- original view and final application;
- synchronized geometry diagnostics;
- a clear success and informative failure or recovery;
- one before/after intervention that improves behavior;
- a concise assumption statement; and
- one honest statement of what the result does not establish.

Choose a single static preview at least 1280 pixels on its longer side when possible. The artifact must remain understandable without code. It may not exceed 75 MB, 720p, 30 fps, or 30 seconds. H.264 uses yuv420p.

## 9. Part E — EXPLORE: Optional extension — up to 10 points

Extend, improve, break, compare, or apply the method in a technically meaningful way. Examples include essential/fundamental geometry, epipolar visualization, calibrated stereo, sparse reconstruction, learned optical flow, inverse-compositional alignment, parallax detection, or piecewise-planar tracking.

- **+5:** meaningful additional exploration with a clear result and explanation.
- **+10:** especially thoughtful, technically interesting, creative, or ambitious exploration.

Optional affirmative showcase participation earns **+1 within the same 10-point cap**. Voting never changes academic credit. Public posting is never required.

## 10. Submission and formatting

Submit the completed notebook/source, one LaTeX-generated answer PDF, required figures/tables, Hero Result, static preview, and acknowledgments of collaborators, resources, data, weights, and AI assistance. Use `answer_template.tex`; do not change stable labels. Begin each labeled answer on a new page and assign it to the corresponding Gradescope item.

Retain raw captures locally. Do not submit API keys, tokens, identifiable private imagery, or unnecessary raw footage. All required code must run on ordinary Colab course hardware.
