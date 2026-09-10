# 1. PA3 data provenance

## 2. Deterministic synthetic data

`setup_data.py --synthetic` generates the required synthetic numerical test data locally from seed `16720` using `assignment3/common/pa3_utils.py`.

Generated cases include:

- 100-point 640×480 homography correspondences with one-pixel Gaussian noise;
- outlier fractions 0, 0.2, 0.4, and 0.6 for seeds 16720–16724;
- textured translation pairs for zero, one-pixel, and larger displacement checks; and
- an additive-brightness pair used to demonstrate violation of B4's `beta=0` assumption.

No external data or personal information is used. Generated files live below `data/generated/`; the starter notebook recreates the tree on first run.

The notebooks and tests call `make_homography_case` and `make_translation_pair` directly rather than loading these files. The `.npz` copies exist so the exact cases can be inspected, diffed, or reloaded. They support implementation and offline smoke checks; they do not replace the graded B4 `wall` row.

## 3. Required B4 sequence and optional debug sequence

Graded B4 requires the `wall` planar image set from the University of Oxford Visual Geometry Group's Affine Covariant Features evaluation data. Run `setup_data.py --b4` to fetch only this required asset. The `graf` set is optional debugging/evaluation data. For backward compatibility, `setup_data.py --external` fetches both sets.

- Project page: <https://www.robots.ox.ac.uk/~vgg/research/affine/>
- `graf`: <https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz>
- `wall`: <https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/wall.tar.gz>

Each archive contains six PPM images and reference homographies from image 1. Every download is SHA-256 verified before extraction:

| Archive | Role | SHA-256 |
|---|---|---|
| `wall.tar.gz` | Required for graded B4 | `3cb3df75f40471f093adf6647d13df6a9f4312f7a222fc591d03f6c08e086344` |
| `graf.tar.gz` | Optional/debug | `999871b945ee968a00a0d5f9af957d1382fb9dae1511cdee9553366817b53b5b` |

The repository does not redistribute these archives. They are downloaded from the publisher for instructional evaluation. Review the publisher's current terms before any redistribution. The required B4 experiment uses `wall/img1.ppm` to `wall/img2.ppm` with `wall/H1to2p`; `graf` remains optional. Neither set satisfies Parts C/D.

## 4. Self-captured student data and approved fallback

Parts C/D require footage captured by the student for this assignment. Students retain raw media locally and submit only the requested compressed evidence. They must avoid non-consenting people, private documents, personal screens, unsafe behavior, and prohibited locations.

The release includes a concrete course-authored alternative under `data/fallback/`. It requires advance, written course-staff approval and is limited to access, privacy, safety, or accessibility barriers. The waiver covers personal recording only; the controlled-condition, prediction, manual-annotation, metrics, implementation, analysis, Hero Result, reflection, and disclosure workload is unchanged. `data/fallback/README.md` defines the request procedure and approval-ID format.

`assignment3/data/generate_fallback.py` procedurally creates the target, background, projective motion, occluder, and illumination changes from seed `16720`; no third-party visual content or personal information is used. Four 6-second, 512×288, 15-fps H.264/yuv420p clips provide baseline, fast-motion, occlusion, and illumination conditions. `ground_truth.json` records every target-to-frame homography and ordered corner, while `manifest.json` fixes byte sizes and SHA-256 hashes. `LICENSE.md` contains the Course Educational Asset License 1.0. Regenerate and verify the release payload with `setup_data.py --fallback`.

## 5. Software-generated visual assets

The default diagnostic overlays are drawn by course helper code and contain no third-party imagery. Students must acknowledge any optional overlay art, pretrained matcher, or learned-flow weights they introduce in Parts C–E.
