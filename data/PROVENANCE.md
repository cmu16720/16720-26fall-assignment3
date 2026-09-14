# 1. PA3 data provenance

## 2. Synthetic data

`setup_data.py --synthetic` generates the synthetic test data locally from seed `16720`, using `pa3_utils.py` in the `common` folder.

The generated cases are:

- 100 homography correspondences in a 640×480 image, with one pixel of Gaussian noise,
- outlier fractions 0, 0.2, 0.4, and 0.6 for seeds 16720 to 16724,
- textured translation pairs for zero, one-pixel, and larger shifts,
- an additive brightness pair that breaks B4's `beta=0` assumption.

No outside data or personal information is used. Generated files are saved in `data/generated/`, and the starter notebook recreates them on its first run.

The notebooks and tests call `make_homography_case` and `make_translation_pair` directly instead of loading these files. The `.npz` copies are there so you can inspect, compare, or reload the exact cases. They help with implementation and offline smoke checks, but they do not replace the graded B4 `wall` row.

## 3. Required B4 sequence and optional debugging sequence

Graded B4 uses the `wall` planar image set from the Affine Covariant Features data of the Visual Geometry Group at the University of Oxford. Run `setup_data.py --b4` to download only this set. The `graf` set is optional and only for debugging. `setup_data.py --external` downloads both sets.

- Project page: <https://www.robots.ox.ac.uk/~vgg/research/affine/>
- `graf`: <https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz>
- `wall`: <https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/wall.tar.gz>

Each archive has six PPM images and reference homographies from image 1. Every download is checked against its SHA-256 hash before extraction:

| Archive | Role | SHA-256 |
|---|---|---|
| `wall.tar.gz` | Required for graded B4 | `3cb3df75f40471f093adf6647d13df6a9f4312f7a222fc591d03f6c08e086344` |
| `graf.tar.gz` | Optional, for debugging | `999871b945ee968a00a0d5f9af957d1382fb9dae1511cdee9553366817b53b5b` |

The repository does not include these archives. They are downloaded from the publisher for course use. Check the publisher's current terms before sharing them further. The required B4 experiment uses `wall/img1.ppm` to `wall/img2.ppm` with `wall/H1to2p`. Neither set counts for Parts C or D.

## 4. Student footage and the approved fallback

Parts C and D need footage that students record themselves for this assignment. Students keep raw videos on their own machines and submit only the requested compressed files. They must not record people who have not agreed to it, private documents, personal screens, unsafe behavior, or places where recording is not allowed.

The course-made package in `data/fallback/` is the only alternative. It needs written approval from course staff in advance, and only for an access, privacy, safety, or accessibility barrier. Approval only replaces the recording. Everything else in Parts C and D is still required. `data/fallback/README.md` explains how to ask and the approval ID format.

`generate_fallback.py` in the `data` folder creates the target, background, perspective motion, occluder, and lighting changes from seed `16720`. No third-party images or personal information are used. Four 6-second, 512×288, 15 fps H.264/yuv420p clips cover the baseline, fast motion, occlusion, and illumination conditions. `ground_truth.json` stores every target-to-frame homography and the ordered corners. `manifest.json` stores the size and SHA-256 hash of every file. `LICENSE.md` contains the Course Educational Asset License 1.0. To regenerate and check the package, run `setup_data.py --fallback`.

## 5. Generated overlay graphics

The default diagnostic overlays are drawn by course helper code and contain no third-party images. Students must acknowledge any overlay art, pretrained matcher, or learned flow weights they add in Parts C, D, or E.
