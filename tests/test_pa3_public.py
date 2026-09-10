"""Public PA3 contract tests distributed with the student starter.

The release defaults to ``student/pa3.py``. Course infrastructure may set the
neutral ``PA3_IMPLEMENTATION_PATH`` environment variable to grade a copied
submission without editing this test file.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing: dataclasses resolves annotations through
    # sys.modules, so a module loaded purely by path would fail to define one.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Loaded by path rather than package so the same file works from the staff tree
# and from the flat student repository.
utils = _load("pa3_public_utils", RELEASE_ROOT / "common" / "pa3_utils.py")
corner_transfer_rmse = utils.corner_transfer_rmse
make_homography_case = utils.make_homography_case
make_translation_pair = utils.make_translation_pair
project_points = utils.project_points

IMPLEMENTATION_PATH = Path(
    os.environ.get("PA3_IMPLEMENTATION_PATH", RELEASE_ROOT / "student" / "pa3.py")
)
pa3 = _load("pa3_public_candidate", IMPLEMENTATION_PATH)


def test_normalize_square_has_released_values() -> None:
    points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    original = points.copy()
    normalized, transform = pa3.normalize_points(points)
    expected = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])
    np.testing.assert_allclose(normalized, expected, atol=1e-12)
    np.testing.assert_allclose(transform, [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], atol=1e-12)
    np.testing.assert_array_equal(points, original)


def test_minimal_dlt_recovers_non_symmetric_homography_without_transpose() -> None:
    source = np.array([[20.0, 15.0], [610.0, 20.0], [600.0, 450.0], [25.0, 460.0]])
    true_h = np.array([[1.03, 0.14, 27.0], [-0.08, 0.96, 19.0], [2.2e-4, -1.4e-4, 1.0]])
    destination, valid = project_points(true_h, source)
    assert valid.all()
    estimated = pa3.estimate_homography_dlt(source, destination)
    np.testing.assert_allclose(estimated / estimated[2, 2], true_h, atol=1e-9, rtol=1e-9)


def test_symmetric_error_is_mean_squared_and_scale_invariant() -> None:
    source = np.array([[3.0, 4.0], [10.0, 12.0], [20.0, 8.0]])
    destination = source + np.array([2.0, -1.0])
    identity = np.eye(3)
    errors = pa3.symmetric_transfer_error(identity, source, destination)
    np.testing.assert_allclose(errors, np.full(3, 5.0), atol=1e-12)
    np.testing.assert_allclose(pa3.symmetric_transfer_error(7.0 * identity, source, destination), errors, atol=1e-12)


def test_ransac_is_deterministic_and_accurate_on_released_case() -> None:
    case = make_homography_case(seed=16720, outlier_fraction=0.4)
    outputs = [
        pa3.ransac_homography(case["src_xy"], case["dst_xy"], 3.0, 0.99, np.random.default_rng(16720))
        for _ in range(2)
    ]
    first_h, first_mask, first_info = outputs[0]
    second_h, second_mask, second_info = outputs[1]
    np.testing.assert_allclose(first_h, second_h, atol=1e-12)
    np.testing.assert_array_equal(first_mask, second_mask)
    assert first_info["trials"] == second_info["trials"] <= pa3.RANSAC_MAX_TRIALS
    assert first_info["best_sample"] == second_info["best_sample"]
    assert isinstance(first_info["best_sample"], tuple) and len(first_info["best_sample"]) == 4
    assert first_info["adaptive_budget"] >= 1
    errors = np.asarray(first_info["errors"], dtype=np.float64)
    assert errors.shape == (100,)
    np.testing.assert_allclose(
        errors, pa3.symmetric_transfer_error(first_h, case["src_xy"], case["dst_xy"]), atol=1e-9
    )
    assert first_mask.dtype == np.bool_ and first_mask.shape == (100,)
    assert corner_transfer_rmse(first_h, case["H_true"], case["image_size"]) < 3.0


def test_four_level_lk_recovers_large_translation() -> None:
    template, image, true_p = make_translation_pair((8.0, -5.0), seed=16722)
    estimate, history = pa3.lucas_kanade_refine(template, image, np.zeros(2), 4)
    np.testing.assert_allclose(estimate, true_p, atol=0.15)
    assert history[0]["phase"] == "level_start" and history[0]["level"] == 3
    assert history[-1]["phase"] == "level_end" and history[-1]["level"] == 0
    required = {"phase", "level", "p_level", "p_full", "status"}
    assert all(required <= set(entry) for entry in history)
    assert {entry["phase"] for entry in history} <= {"level_start", "iteration", "level_end"}
    iterations = [entry for entry in history if entry["phase"] == "iteration"]
    assert iterations and all("photometric_rmse" in entry for entry in iterations)
    for entry in history:
        np.testing.assert_allclose(
            entry["p_full"], np.asarray(entry["p_level"]) * 2.0 ** entry["level"], atol=1e-12
        )


def test_released_constants() -> None:
    assert pa3.RANSAC_MAX_TRIALS == 5000
    assert pa3.LK_MAX_ITERS == 50
    assert pa3.LK_TOL == pytest.approx(1e-3)
    assert pa3.LK_MIN_VALID == 25
    assert pa3.LK_MAX_CONDITION == pytest.approx(1e6)
