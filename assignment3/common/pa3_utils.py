"""Supplied geometry, image, data, visualization, and video helpers for PA3.

The required student algorithms live in ``assignment3/student/pa3.py``.  This
module deliberately supplies infrastructure that is useful but is not the
learning objective: coordinate conversion, inverse image warping, bilinear
sampling, pyramids, deterministic generators, SIFT matching, diagnostics, and
artifact I/O.

Coordinate convention
---------------------
Points have shape ``(N, 2)`` and order ``(x, y)``.  Images are indexed
``image[y, x]``.  Homographies map source points to destination points even
though image warping samples in the inverse direction.  Geometry arrays are
``float64`` and inputs are not mutated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


EPS = 1e-12
SHARED_SEED = 16720


def validate_xy(xy: np.ndarray, *, min_points: int = 0, name: str = "xy") -> np.ndarray:
    """Return a validated float64 copy of an ``(N, 2)`` point array."""

    array = np.asarray(xy, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2); received {array.shape}")
    if array.shape[0] < min_points:
        raise ValueError(f"{name} must contain at least {min_points} points")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def to_homogeneous(xy: np.ndarray) -> np.ndarray:
    """Convert ``(N, 2)`` Cartesian points to ``(N, 3)`` homogeneous rows."""

    points = validate_xy(xy)
    return np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))


def from_homogeneous(xyh: np.ndarray, *, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    """Convert homogeneous rows to Cartesian points and an explicit valid mask.

    Invalid output rows are filled with ``nan``.  A row is valid only when all
    coordinates are finite and the absolute homogeneous scale exceeds ``eps``.
    """

    points = np.asarray(xyh, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"xyh must have shape (N, 3); received {points.shape}")
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    valid = np.all(np.isfinite(points), axis=1) & (np.abs(points[:, 2]) > eps)
    output = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    output[valid] = points[valid, :2] / points[valid, 2, None]
    return output, valid


def project_points(H: np.ndarray, xy: np.ndarray, *, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    """Project Cartesian source points through a source-to-destination homography."""

    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("H must be a finite 3 x 3 matrix")
    projected_h = (matrix @ to_homogeneous(xy).T).T
    return from_homogeneous(projected_h, eps=eps)


def bilinear_sample(image: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample a grayscale or channel-last image at fractional ``(x, y)`` points.

    Returns sampled float64 values and a Boolean validity mask.  Invalid samples
    are ``nan`` rather than silently clipped to the border.
    """

    array = np.asarray(image, dtype=np.float64)
    if array.ndim not in (2, 3) or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("image must have shape (H, W) or (H, W, C) with H,W > 0")
    points = validate_xy(xy)
    height, width = array.shape[:2]
    valid = (
        np.all(np.isfinite(points), axis=1)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= height - 1)
    )
    value_shape = (points.shape[0],) if array.ndim == 2 else (points.shape[0], array.shape[2])
    values = np.full(value_shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return values, valid

    x = points[valid, 0]
    y = points[valid, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0

    if array.ndim == 3:
        wx = wx[:, None]
        wy = wy[:, None]
    v00 = array[y0, x0]
    v10 = array[y0, x1]
    v01 = array[y1, x0]
    v11 = array[y1, x1]
    values[valid] = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )
    return values, valid


def warp_image(
    image: np.ndarray,
    H_src_to_dst: np.ndarray,
    output_shape: tuple[int, int],
    *,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-sample ``image`` into a destination canvas.

    ``H_src_to_dst`` reports the forward geometry.  The implementation solves
    for the source coordinate of every destination pixel and then calls
    :func:`bilinear_sample`.  The returned mask identifies valid output pixels.
    """

    matrix = np.asarray(H_src_to_dst, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("H_src_to_dst must be a finite 3 x 3 matrix")
    if len(output_shape) != 2 or min(output_shape) <= 0:
        raise ValueError("output_shape must be a positive (height, width) pair")
    height, width = (int(output_shape[0]), int(output_shape[1]))
    yy, xx = np.indices((height, width), dtype=np.float64)
    dst_h = np.column_stack((xx.ravel(), yy.ravel(), np.ones(height * width)))
    try:
        src_h = np.linalg.solve(matrix, dst_h.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("H_src_to_dst must be invertible") from exc
    src_xy, homogeneous_valid = from_homogeneous(src_h)
    sampled, sample_valid = bilinear_sample(image, src_xy)
    valid = homogeneous_valid & sample_valid
    source = np.asarray(image)
    result_shape = (height, width) if source.ndim == 2 else (height, width, source.shape[2])
    result = np.full(result_shape, fill_value, dtype=np.float64)
    if source.ndim == 2:
        result.reshape(-1)[valid] = sampled[valid]
    else:
        result.reshape(-1, source.shape[2])[valid] = sampled[valid]
    return result, valid.reshape(height, width)


def image_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return supplied central-difference gradients ``(Ix, Iy)``."""

    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2 or not np.all(np.isfinite(array)):
        raise ValueError("image must be a finite grayscale array with each dimension >= 2")
    grad_y, grad_x = np.gradient(array)
    return grad_x, grad_y


def _gaussian_blur5(image: np.ndarray) -> np.ndarray:
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    array = np.asarray(image, dtype=np.float64)
    pad_width = ((0, 0), (2, 2)) if array.ndim == 2 else ((0, 0), (2, 2), (0, 0))
    horizontal_input = np.pad(array, pad_width, mode="reflect")
    horizontal = sum(kernel[index] * horizontal_input[:, index : index + array.shape[1]] for index in range(5))
    pad_width = ((2, 2), (0, 0)) if array.ndim == 2 else ((2, 2), (0, 0), (0, 0))
    vertical_input = np.pad(horizontal, pad_width, mode="reflect")
    return sum(kernel[index] * vertical_input[index : index + array.shape[0], :] for index in range(5))


def gaussian_pyramid(image: np.ndarray, levels: int = 4) -> list[np.ndarray]:
    """Build a level-0-full-resolution Gaussian pyramid using a supplied filter."""

    array = np.asarray(image, dtype=np.float64)
    if array.ndim not in (2, 3) or min(array.shape[:2]) < 2:
        raise ValueError("image must be nonempty grayscale or channel-last data")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must be finite")
    if not isinstance(levels, (int, np.integer)) or levels < 1:
        raise ValueError("levels must be a positive integer")
    pyramid = [array.copy()]
    for _ in range(1, int(levels)):
        if min(pyramid[-1].shape[:2]) < 4:
            raise ValueError("too many pyramid levels for the image dimensions")
        pyramid.append(_gaussian_blur5(pyramid[-1])[::2, ::2].copy())
    return pyramid


def make_textured_image(shape: tuple[int, int] = (128, 160), *, seed: int = SHARED_SEED) -> np.ndarray:
    """Create a deterministic, band-limited grayscale texture in ``[0.1, 0.9]``."""

    height, width = (int(shape[0]), int(shape[1]))
    if min(height, width) < 16:
        raise ValueError("shape dimensions must both be at least 16")
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(height, width))
    for _ in range(3):
        noise = _gaussian_blur5(noise)
    yy, xx = np.indices((height, width), dtype=np.float64)
    # Random band-limited structure dominates; weak sinusoids keep gradients
    # present near borders without making the texture strongly repetitive.
    # This yields a released large-motion case where a pyramid materially
    # expands the convergence basin instead of both methods locking to a
    # periodic false minimum.
    pattern = 2.0 * noise + 0.15 * np.sin(xx / 7.0) + 0.10 * np.cos(yy / 9.0) + 0.05 * np.sin((xx + yy) / 13.0)
    pattern -= pattern.min()
    denominator = max(float(pattern.max()), EPS)
    return 0.1 + 0.8 * pattern / denominator


def make_translation_pair(
    shift: Sequence[float] = (8.0, -5.0),
    *,
    shape: tuple[int, int] = (128, 160),
    beta: float = 0.0,
    seed: int = SHARED_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create ``template, image, true_p`` satisfying ``image(x+p)=template(x)+beta``."""

    p = np.asarray(shift, dtype=np.float64)
    if p.shape != (2,) or not np.all(np.isfinite(p)) or not np.isfinite(beta):
        raise ValueError("shift must be a finite length-2 vector and beta must be finite")
    template = make_textured_image(shape, seed=seed)
    H = np.array([[1.0, 0.0, p[0]], [0.0, 1.0, p[1]], [0.0, 0.0, 1.0]])
    image, valid = warp_image(template, H, shape, fill_value=0.5)
    image[valid] += float(beta)
    return template, image, p.copy()


def make_homography_case(
    *,
    seed: int = SHARED_SEED,
    n_points: int = 100,
    image_size: tuple[int, int] = (640, 480),
    noise_std: float = 1.0,
    outlier_fraction: float = 0.3,
    H_true: np.ndarray | None = None,
) -> dict[str, np.ndarray | tuple[int, int] | float | int]:
    """Generate a deterministic point-correspondence case with known truth."""

    if n_points < 4:
        raise ValueError("n_points must be at least four")
    if noise_std < 0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and nonnegative")
    if not 0.0 <= outlier_fraction < 1.0:
        raise ValueError("outlier_fraction must lie in [0, 1)")
    width, height = (int(image_size[0]), int(image_size[1]))
    if min(width, height) < 64:
        raise ValueError("image_size must be at least 64 x 64")
    if H_true is None:
        matrix = np.array(
            [[1.02, 0.07, 28.0], [-0.035, 1.04, 18.0], [1.5e-4, -1.1e-4, 1.0]],
            dtype=np.float64,
        )
    else:
        matrix = np.asarray(H_true, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) <= EPS:
            raise ValueError("H_true must be a finite invertible 3 x 3 matrix")
        matrix = matrix.copy()
    rng = np.random.default_rng(seed)
    margin = 0.08
    src = np.column_stack(
        (
            rng.uniform(margin * width, (1.0 - margin) * width, n_points),
            rng.uniform(margin * height, (1.0 - margin) * height, n_points),
        )
    )
    dst_clean, valid = project_points(matrix, src)
    if not np.all(valid):
        raise RuntimeError("the deterministic homography unexpectedly projected an invalid point")
    dst = dst_clean + rng.normal(0.0, noise_std, dst_clean.shape)
    n_outliers = int(round(outlier_fraction * n_points))
    true_inliers = np.ones(n_points, dtype=bool)
    if n_outliers:
        outlier_indices = rng.choice(n_points, n_outliers, replace=False)
        true_inliers[outlier_indices] = False
        dst[outlier_indices, 0] = rng.uniform(0.0, width - 1.0, n_outliers)
        dst[outlier_indices, 1] = rng.uniform(0.0, height - 1.0, n_outliers)
    permutation = rng.permutation(n_points)
    return {
        "src_xy": src[permutation],
        "dst_xy": dst[permutation],
        "dst_clean_xy": dst_clean[permutation],
        "true_inliers": true_inliers[permutation],
        "H_true": matrix,
        "image_size": (width, height),
        "noise_std": float(noise_std),
        "seed": int(seed),
    }


def image_corners(image_size: tuple[int, int]) -> np.ndarray:
    """Return clockwise image corners for an ``(width, height)`` size."""

    width, height = image_size
    return np.array([[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]])


def corner_transfer_rmse(H_est: np.ndarray, H_true: np.ndarray, image_size: tuple[int, int]) -> float:
    """Compute scale-invariant RMS disagreement on four image corners."""

    corners = image_corners(image_size)
    estimated, valid_est = project_points(H_est, corners)
    expected, valid_true = project_points(H_true, corners)
    if not np.all(valid_est & valid_true):
        return float("inf")
    return float(np.sqrt(np.mean(np.sum((estimated - expected) ** 2, axis=1))))


def inlier_precision_recall(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Return precision, recall, and F1 for Boolean inlier masks."""

    predicted_mask = np.asarray(predicted, dtype=bool)
    truth_mask = np.asarray(truth, dtype=bool)
    if predicted_mask.shape != truth_mask.shape or predicted_mask.ndim != 1:
        raise ValueError("predicted and truth must be same-shape one-dimensional masks")
    true_positive = int(np.count_nonzero(predicted_mask & truth_mask))
    precision = true_positive / max(int(np.count_nonzero(predicted_mask)), 1)
    recall = true_positive / max(int(np.count_nonzero(truth_mask)), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise convex hull using deterministic monotone chains."""

    xy = validate_xy(points)
    if len(xy) <= 1:
        return xy
    unique = np.unique(xy, axis=0)
    if len(unique) <= 2:
        return unique

    def cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))

    lower: list[np.ndarray] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in unique[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def polygon_area(polygon: np.ndarray) -> float:
    """Return the unsigned shoelace area of an ordered polygon."""

    points = validate_xy(polygon)
    if len(points) < 3:
        return 0.0
    return float(0.5 * abs(np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1))))


def _signed_polygon_area(points: np.ndarray) -> float:
    return 0.5 * float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1)))


def _convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = np.asarray(subject, dtype=np.float64)
    clipping = np.asarray(clip, dtype=np.float64)
    if len(output) < 3 or len(clipping) < 3:
        return np.empty((0, 2), dtype=np.float64)
    if _signed_polygon_area(clipping) < 0:
        clipping = clipping[::-1]

    def inside(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
        return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= -1e-10

    def intersection(start: np.ndarray, end: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        direction = end - start
        edge = b - a
        denominator = direction[0] * edge[1] - direction[1] * edge[0]
        if abs(denominator) <= EPS:
            return end.copy()
        delta = a - start
        t = (delta[0] * edge[1] - delta[1] * edge[0]) / denominator
        return start + t * direction

    for index in range(len(clipping)):
        clip_a = clipping[index]
        clip_b = clipping[(index + 1) % len(clipping)]
        input_polygon = output
        output_list: list[np.ndarray] = []
        if len(input_polygon) == 0:
            break
        start = input_polygon[-1]
        for end in input_polygon:
            end_inside = inside(end, clip_a, clip_b)
            start_inside = inside(start, clip_a, clip_b)
            if end_inside:
                if not start_inside:
                    output_list.append(intersection(start, end, clip_a, clip_b))
                output_list.append(end)
            elif start_inside:
                output_list.append(intersection(start, end, clip_a, clip_b))
            start = end
        output = np.asarray(output_list, dtype=np.float64).reshape(-1, 2)
    return output


def quadrilateral_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return intersection-over-union for two convex quadrilaterals."""

    first = validate_xy(a)
    second = validate_xy(b)
    if first.shape != (4, 2) or second.shape != (4, 2):
        raise ValueError("a and b must each have shape (4, 2)")
    first_hull = convex_hull(first)
    second_hull = convex_hull(second)
    intersection = _convex_intersection(first_hull, second_hull)
    intersection_area = polygon_area(intersection)
    union_area = polygon_area(first_hull) + polygon_area(second_hull) - intersection_area
    return float(intersection_area / union_area) if union_area > EPS else 0.0


def manual_corner_metrics(predicted: np.ndarray, annotated: np.ndarray) -> dict[str, float]:
    """Measure ordered-corner error and quadrilateral IoU against manual labels."""

    predicted_xy = validate_xy(predicted)
    annotated_xy = validate_xy(annotated)
    if predicted_xy.shape != (4, 2) or annotated_xy.shape != (4, 2):
        raise ValueError("predicted and annotated corners must each have shape (4, 2)")
    errors = np.linalg.norm(predicted_xy - annotated_xy, axis=1)
    return {
        "mean_corner_error_px": float(np.mean(errors)),
        "max_corner_error_px": float(np.max(errors)),
        "quad_iou": quadrilateral_iou(predicted_xy, annotated_xy),
    }


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] not in (3, 4):
            raise ValueError("color image must have three or four channels")
        rgb = array[..., :3].astype(np.float64)
        array = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if array.ndim != 2:
        raise ValueError("image must be grayscale or RGB/RGBA")
    numeric = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("image must be finite")
    if numeric.max(initial=0.0) <= 1.0:
        numeric = numeric * 255.0
    return np.clip(np.rint(numeric), 0, 255).astype(np.uint8)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB image data to display-ready uint8 RGB."""

    array = np.asarray(image)
    if array.ndim == 2:
        gray = _to_gray_uint8(array)
        return np.repeat(gray[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("image must be grayscale, RGB, or RGBA")
    numeric = np.asarray(array[..., :3], dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("image must be finite")
    if numeric.max(initial=0.0) <= 1.0:
        numeric *= 255.0
    return np.clip(np.rint(numeric), 0, 255).astype(np.uint8)


def extract_sift(image: np.ndarray, *, max_features: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Extract supplied SIFT keypoints and descriptors using OpenCV."""

    if max_features < 1:
        raise ValueError("max_features must be positive")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the supplied SIFT baseline") from exc
    detector = cv2.SIFT_create(nfeatures=int(max_features))
    keypoints, descriptors = detector.detectAndCompute(_to_gray_uint8(image), None)
    if not keypoints or descriptors is None:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 128), dtype=np.float32)
    xy = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float64)
    return xy, np.asarray(descriptors, dtype=np.float32)


def match_sift(
    descriptors_src: np.ndarray,
    descriptors_dst: np.ndarray,
    *,
    ratio: float = 0.75,
    mutual: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the supplied L2 two-neighbor ratio test and optional mutual check."""

    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must lie strictly between zero and one")
    src = np.asarray(descriptors_src, dtype=np.float32)
    dst = np.asarray(descriptors_dst, dtype=np.float32)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1:] != dst.shape[1:]:
        raise ValueError("descriptor arrays must have shape (N, D) with matching D")
    if len(src) == 0 or len(dst) < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the supplied SIFT baseline") from exc
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def accepted(query: np.ndarray, train: np.ndarray) -> dict[tuple[int, int], float]:
        output: dict[tuple[int, int], float] = {}
        if len(train) < 2:
            return output
        for pair in matcher.knnMatch(query, train, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                output[(pair[0].queryIdx, pair[0].trainIdx)] = float(pair[0].distance)
        return output

    forward = accepted(src, dst)
    if mutual:
        backward = accepted(dst, src)
        forward = {pair: distance for pair, distance in forward.items() if (pair[1], pair[0]) in backward}
    ordered = sorted(forward.items(), key=lambda item: (item[1], item[0][0], item[0][1]))
    if not ordered:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    pairs = np.asarray([item[0] for item in ordered], dtype=np.int64)
    distances = np.asarray([item[1] for item in ordered], dtype=np.float64)
    return pairs, distances


def sift_correspondences(
    source_image: np.ndarray,
    destination_image: np.ndarray,
    *,
    max_features: int = 2000,
    ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return supplied SIFT source/destination matches and diagnostics."""

    source_xy, source_desc = extract_sift(source_image, max_features=max_features)
    destination_xy, destination_desc = extract_sift(destination_image, max_features=max_features)
    pairs, distances = match_sift(source_desc, destination_desc, ratio=ratio, mutual=True)
    return source_xy[pairs[:, 0]], destination_xy[pairs[:, 1]], {
        "source_keypoints": source_xy,
        "destination_keypoints": destination_xy,
        "pairs": pairs,
        "distances": distances,
    }


def tracking_validity(
    H: np.ndarray,
    source_xy: np.ndarray,
    inlier_mask: np.ndarray,
    symmetric_errors: np.ndarray,
    *,
    target_size: tuple[int, int],
    frame_shape: tuple[int, int],
    threshold_px: float = 3.0,
) -> dict[str, object]:
    """Apply the released confidence and validity gates for the C baseline."""

    points = validate_xy(source_xy)
    mask = np.asarray(inlier_mask, dtype=bool)
    errors = np.asarray(symmetric_errors, dtype=np.float64)
    if mask.shape != (len(points),) or errors.shape != (len(points),):
        raise ValueError("inlier_mask and symmetric_errors must have shape (N,)")
    if threshold_px <= 0 or not np.isfinite(threshold_px):
        raise ValueError("threshold_px must be finite and positive")
    n_matches = len(points)
    n_inliers = int(np.count_nonzero(mask))
    ratio = n_inliers / max(n_matches, 1)
    finite_inlier_errors = errors[mask & np.isfinite(errors)]
    median_rms = float(np.sqrt(np.median(finite_inlier_errors))) if len(finite_inlier_errors) else float("inf")
    target_area = max(float((target_size[0] - 1) * (target_size[1] - 1)), 1.0)
    coverage = polygon_area(convex_hull(points[mask])) / target_area if n_inliers >= 3 else 0.0
    corners, projected_valid = project_points(H, image_corners(target_size))
    frame_area = float(frame_shape[0] * frame_shape[1])
    quad_area = polygon_area(convex_hull(corners)) if np.all(projected_valid) else 0.0
    quad_ok = bool(np.all(projected_valid) and len(convex_hull(corners)) == 4 and 64.0 <= quad_area <= 4.0 * frame_area)
    valid = bool(n_inliers >= 10 and ratio >= 0.25 and coverage >= 0.15 and median_rms <= threshold_px and quad_ok)
    confidence = float(
        np.clip(ratio * np.exp(-((median_rms / threshold_px) ** 2)) * min(1.0, coverage / 0.25), 0.0, 1.0)
    ) if np.isfinite(median_rms) else 0.0
    return {
        "valid": valid,
        "confidence": confidence,
        "n_matches": n_matches,
        "n_inliers": n_inliers,
        "inlier_ratio": float(ratio),
        "median_symmetric_error_px": median_rms,
        "source_coverage": float(coverage),
        "projected_corners": corners,
        "projected_quad_area": float(quad_area),
        "quad_valid": quad_ok,
    }


def draw_correspondences(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    *,
    inlier_mask: np.ndarray | None = None,
    max_draw: int = 100,
) -> np.ndarray:
    """Draw deterministic correspondence diagnostics on a side-by-side RGB canvas."""

    if max_draw < 1:
        raise ValueError("max_draw must be positive")
    first = to_uint8_rgb(image_src)
    second = to_uint8_rgb(image_dst)
    points_src = validate_xy(src_xy)
    points_dst = validate_xy(dst_xy)
    if points_src.shape != points_dst.shape:
        raise ValueError("src_xy and dst_xy must have matching shapes")
    if inlier_mask is None:
        mask = np.ones(len(points_src), dtype=bool)
    else:
        mask = np.asarray(inlier_mask, dtype=bool)
        if mask.shape != (len(points_src),):
            raise ValueError("inlier_mask must have shape (N,)")
    height = max(first.shape[0], second.shape[0])
    canvas = np.zeros((height, first.shape[1] + second.shape[1], 3), dtype=np.uint8)
    canvas[: first.shape[0], : first.shape[1]] = first
    canvas[: second.shape[0], first.shape[1] :] = second
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for supplied visualization") from exc
    indices = np.arange(min(len(points_src), max_draw))
    for index in indices:
        a = tuple(np.rint(points_src[index]).astype(int))
        b_xy = np.rint(points_dst[index]).astype(int)
        b = (int(b_xy[0] + first.shape[1]), int(b_xy[1]))
        color = (30, 210, 70) if mask[index] else (230, 70, 55)
        cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, 2, color, -1, cv2.LINE_AA)
    return canvas


def draw_projected_quad(
    image: np.ndarray,
    H_target_to_frame: np.ndarray,
    target_size: tuple[int, int],
    *,
    color: tuple[int, int, int] = (30, 220, 80),
    thickness: int = 3,
) -> np.ndarray:
    """Draw a homography-projected target boundary on a copy of an image."""

    canvas = to_uint8_rgb(image)
    corners, valid = project_points(H_target_to_frame, image_corners(target_size))
    if not np.all(valid):
        return canvas
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for supplied visualization") from exc
    cv2.polylines(canvas, [np.rint(corners).astype(np.int32)], True, color, int(thickness), cv2.LINE_AA)
    return canvas


def resize_long_edge(image: np.ndarray, max_long_edge: int = 720) -> tuple[np.ndarray, float]:
    """Resize while preserving aspect ratio and return output/source coordinate scale."""

    if max_long_edge < 1:
        raise ValueError("max_long_edge must be positive")
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be grayscale or channel-last")
    scale = min(1.0, max_long_edge / max(array.shape[:2]))
    if scale == 1.0:
        return array.copy(), 1.0
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for supplied resizing") from exc
    size = (max(1, int(round(array.shape[1] * scale))), max(1, int(round(array.shape[0] * scale))))
    return cv2.resize(array, size, interpolation=cv2.INTER_AREA), float(scale)


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata returned by :func:`read_video`."""

    fps: float
    source_width: int
    source_height: int
    coordinate_scale: float
    frame_count: int


def read_video(
    path: str | Path,
    *,
    max_frames: int | None = None,
    max_long_edge: int = 720,
) -> tuple[list[np.ndarray], VideoMetadata]:
    """Decode an RGB video and record the resize transform."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for supplied video decoding") from exc
    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    scale = 1.0
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb, scale = resize_long_edge(rgb, max_long_edge=max_long_edge)
            frames.append(rgb)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video contained no decodable frames: {source}")
    return frames, VideoMetadata(fps, width, height, scale, len(frames))


def write_h264_video(path: str | Path, frames: Iterable[np.ndarray], *, fps: float = 30.0) -> Path:
    """Write RGB frames as H.264/yuv420p MP4 using imageio-ffmpeg."""

    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be finite and positive")
    frame_list = [to_uint8_rgb(frame) for frame in frames]
    if not frame_list:
        raise ValueError("frames must be nonempty")
    shape = frame_list[0].shape
    if any(frame.shape != shape for frame in frame_list):
        raise ValueError("all frames must have the same shape")
    if shape[0] > 720 or shape[1] > 1280:
        raise ValueError("PA3 artifacts may not exceed 720p")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio and imageio-ffmpeg are required for H.264 export") from exc
    with imageio.get_writer(
        destination,
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        ffmpeg_log_level="warning",
    ) as writer:
        for frame in frame_list:
            writer.append_data(frame)
    return destination


def plot_tracking_trace(
    confidence: Sequence[float],
    error_px: Sequence[float],
    *,
    output_path: str | Path | None = None,
):
    """Create the supplied synchronized confidence/error trace figure."""

    import matplotlib.pyplot as plt

    confidence_array = np.asarray(confidence, dtype=np.float64)
    error_array = np.asarray(error_px, dtype=np.float64)
    if confidence_array.shape != error_array.shape or confidence_array.ndim != 1:
        raise ValueError("confidence and error_px must be same-shape one-dimensional arrays")
    figure, first_axis = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    frames = np.arange(len(confidence_array))
    first_axis.plot(frames, confidence_array, color="tab:green", label="confidence")
    first_axis.set(xlabel="frame", ylabel="confidence", ylim=(-0.02, 1.02))
    second_axis = first_axis.twinx()
    second_axis.plot(frames, error_array, color="tab:red", alpha=0.8, label="corner error")
    second_axis.set_ylabel("error (px)")
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
    return figure


__all__ = [
    "EPS",
    "SHARED_SEED",
    "VideoMetadata",
    "bilinear_sample",
    "corner_transfer_rmse",
    "convex_hull",
    "draw_correspondences",
    "draw_projected_quad",
    "extract_sift",
    "from_homogeneous",
    "gaussian_pyramid",
    "image_corners",
    "image_gradients",
    "inlier_precision_recall",
    "make_homography_case",
    "make_textured_image",
    "make_translation_pair",
    "manual_corner_metrics",
    "match_sift",
    "plot_tracking_trace",
    "polygon_area",
    "project_points",
    "quadrilateral_iou",
    "read_video",
    "resize_long_edge",
    "sift_correspondences",
    "to_homogeneous",
    "to_uint8_rgb",
    "tracking_validity",
    "validate_xy",
    "warp_image",
    "write_h264_video",
]
