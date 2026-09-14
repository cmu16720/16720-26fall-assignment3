"""Generate and verify the staff-approved PA3 accessibility fallback package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator

import numpy as np


FALLBACK_VERSION = "1.0"
SHARED_SEED = 16720
FRAME_WIDTH = 512
FRAME_HEIGHT = 288
FPS = 15
FRAME_COUNT = 90
DURATION_SECONDS = FRAME_COUNT / FPS
TARGET_WIDTH = 192
TARGET_HEIGHT = 128
CONDITIONS = ("baseline", "fast_motion", "occlusion", "illumination")
EVALUATION_FRAMES = tuple(range(0, FRAME_COUNT, 10))
DEFAULT_FALLBACK_ROOT = Path(__file__).resolve().parent / "fallback"
POLICY_FILES = ("README.md", "LICENSE.md")


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to regenerate the PA3 fallback package") from exc
    cv2.setNumThreads(1)
    return cv2


def make_planar_target() -> np.ndarray:
    """Return the deterministic, course-made RGB target image."""

    cv2 = _opencv()
    yy, xx = np.indices((TARGET_HEIGHT, TARGET_WIDTH))
    checker = ((xx // 16 + yy // 16) % 2).astype(np.uint8)
    target = np.empty((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    target[checker == 0] = (244, 238, 218)
    target[checker == 1] = (32, 48, 76)

    rng = np.random.default_rng(SHARED_SEED)
    palette = ((230, 67, 80), (30, 170, 145), (247, 180, 45), (80, 105, 225))
    for index in range(28):
        center = tuple(int(value) for value in rng.integers((8, 8), (TARGET_WIDTH - 8, TARGET_HEIGHT - 8)))
        radius = int(rng.integers(2, 7))
        cv2.circle(target, center, radius, palette[index % len(palette)], -1, lineType=cv2.LINE_AA)

    cv2.rectangle(target, (4, 4), (TARGET_WIDTH - 5, TARGET_HEIGHT - 5), (255, 255, 255), 3)
    cv2.rectangle(target, (11, 43), (TARGET_WIDTH - 12, 86), (8, 12, 24), -1)
    cv2.putText(target, "16-720", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(target, "PLANAR", (50, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (247, 180, 45), 1, cv2.LINE_AA)
    for point in ((14, 14), (TARGET_WIDTH - 15, 14), (TARGET_WIDTH - 15, TARGET_HEIGHT - 15), (14, TARGET_HEIGHT - 15)):
        cv2.drawMarker(target, point, (255, 70, 70), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
    return target


def _background() -> np.ndarray:
    yy, xx = np.indices((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float64)
    base = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float64)
    base[..., 0] = 48 + 28 * xx / (FRAME_WIDTH - 1)
    base[..., 1] = 62 + 24 * yy / (FRAME_HEIGHT - 1)
    base[..., 2] = 78 + 18 * (xx + yy) / (FRAME_WIDTH + FRAME_HEIGHT - 2)
    grid = ((xx.astype(np.int64) // 32 + yy.astype(np.int64) // 32) % 2) * 7
    return np.clip(base + grid[..., None], 0, 255).astype(np.uint8)


def _target_corners(frame_index: int, condition: str) -> np.ndarray:
    progress = frame_index / (FRAME_COUNT - 1)
    phase = 2.0 * np.pi * progress
    rate = 2.75 if condition == "fast_motion" else 1.0
    moving_phase = rate * phase
    center = np.array(
        [
            FRAME_WIDTH / 2 + 58 * np.sin(moving_phase),
            FRAME_HEIGHT / 2 + 25 * np.sin(0.8 * moving_phase + 0.6),
        ],
        dtype=np.float64,
    )
    half_width = 109 * (1.0 + 0.06 * np.cos(0.7 * moving_phase))
    half_height = 73 * (1.0 + 0.05 * np.sin(0.9 * moving_phase))
    angle = 0.11 * np.sin(0.65 * moving_phase)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local = np.array(
        [[-half_width, -half_height], [half_width, -half_height], [half_width, half_height], [-half_width, half_height]],
        dtype=np.float64,
    )
    corners = local @ rotation.T + center
    perspective = 11 * np.sin(0.55 * moving_phase + 0.4)
    corners += np.array([[-perspective, 5], [perspective, -5], [-0.45 * perspective, 4], [0.45 * perspective, -4]])
    return corners


def _frame_record(frame_index: int, condition: str, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cv2 = _opencv()
    source_corners = np.array(
        [[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]],
        dtype=np.float32,
    )
    destination_corners = _target_corners(frame_index, condition)
    homography = cv2.getPerspectiveTransform(source_corners, destination_corners.astype(np.float32)).astype(np.float64)
    canvas = _background()
    warped = cv2.warpPerspective(target, homography, (FRAME_WIDTH, FRAME_HEIGHT), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.full((TARGET_HEIGHT, TARGET_WIDTH), 255, dtype=np.uint8),
        homography,
        (FRAME_WIDTH, FRAME_HEIGHT),
        flags=cv2.INTER_NEAREST,
    )
    canvas[mask > 0] = warped[mask > 0]

    occlusion_fraction = 0.0
    if condition == "occlusion" and 25 <= frame_index <= 64:
        occluder_source = np.array(
            [
                [0.20 * TARGET_WIDTH, 0.02 * TARGET_HEIGHT],
                [1.02 * TARGET_WIDTH, 0.02 * TARGET_HEIGHT],
                [1.02 * TARGET_WIDTH, 0.98 * TARGET_HEIGHT],
                [0.20 * TARGET_WIDTH, 0.98 * TARGET_HEIGHT],
            ],
            dtype=np.float32,
        )
        occluder = cv2.perspectiveTransform(occluder_source[None, :, :], homography)[0]
        cv2.fillConvexPoly(canvas, np.round(occluder).astype(np.int32), (35, 45, 58), lineType=cv2.LINE_AA)
        occlusion_fraction = float((1.0 - 0.20) * (0.98 - 0.02))

    brightness_gain = 1.0
    brightness_bias = 0.0
    if condition == "illumination":
        brightness_gain = float(0.68 + 0.52 * (0.5 + 0.5 * np.sin(2.0 * np.pi * frame_index / 45.0)))
        brightness_bias = float(8.0 * np.sin(2.0 * np.pi * frame_index / 30.0))
        canvas = np.clip(canvas.astype(np.float64) * brightness_gain + brightness_bias, 0, 255).astype(np.uint8)

    record = {
        "frame_index": frame_index,
        "time_seconds": frame_index / FPS,
        "H_target_to_frame": homography.tolist(),
        "corners_xy": destination_corners.tolist(),
        "brightness_gain": brightness_gain,
        "brightness_bias": brightness_bias,
        "occlusion_fraction_target": occlusion_fraction,
    }
    return canvas, record


def fallback_clip_frames(condition: str) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    """Yield deterministic RGB frames and ground truth for one condition."""

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    target = make_planar_target()
    for frame_index in range(FRAME_COUNT):
        yield _frame_record(frame_index, condition, target)


def _encode_h264(path: Path, frames: list[np.ndarray]) -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is required to encode the fallback clips") from exc
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        executable,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-map_metadata",
        "-1",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "27",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    payload = np.stack(frames).astype(np.uint8, copy=False).tobytes(order="C")
    completed = subprocess.run(command, input=payload, capture_output=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"fallback H.264 encoding failed: {message}")
    return imageio_ffmpeg.get_ffmpeg_version()


def _copy_policy_files(output_root: Path) -> None:
    canonical_root = DEFAULT_FALLBACK_ROOT.resolve()
    destination_root = output_root.resolve()
    for filename in POLICY_FILES:
        source = canonical_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"missing canonical fallback policy file: {source}")
        destination = destination_root / filename
        if destination != source:
            shutil.copy2(source, destination)


def generate_fallback_package(output_root: Path = DEFAULT_FALLBACK_ROOT) -> list[Path]:
    """Regenerate every media/metadata asset and its SHA-256 manifest."""

    cv2 = _opencv()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_policy_files(output_root)

    target_path = output_root / "planar_target.png"
    target_rgb = make_planar_target()
    success, encoded_target = cv2.imencode(".png", cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("could not encode the fallback planar target")
    target_path.write_bytes(encoded_target.tobytes())

    clips: dict[str, Any] = {}
    written = [output_root / filename for filename in POLICY_FILES] + [target_path]
    encoder_version = ""
    for condition in CONDITIONS:
        frames: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        raw_digest = hashlib.sha256()
        for frame, record in fallback_clip_frames(condition):
            frames.append(frame)
            records.append(record)
            raw_digest.update(frame.tobytes(order="C"))
        clip_path = output_root / f"{condition}.mp4"
        encoder_version = _encode_h264(clip_path, frames)
        written.append(clip_path)
        clips[condition] = {
            "file": clip_path.name,
            "changed_factor": {
                "baseline": "none",
                "fast_motion": "inter-frame displacement only",
                "occlusion": "target occlusion only",
                "illumination": "global brightness gain/bias only",
            }[condition],
            "raw_rgb_sha256": raw_digest.hexdigest(),
            "frames": records,
        }

    ground_truth = {
        "schema_version": 1,
        "package_version": FALLBACK_VERSION,
        "authorship": "course-made procedural imagery with no third-party visual content",
        "seed": SHARED_SEED,
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "fps": FPS,
        "frame_count": FRAME_COUNT,
        "duration_seconds": DURATION_SECONDS,
        "target_size": [TARGET_WIDTH, TARGET_HEIGHT],
        "corner_order": "top-left, top-right, bottom-right, bottom-left in (x,y)",
        "evaluation_frames": list(EVALUATION_FRAMES),
        "student_requirement": "hand-marked corners are still required, and these values are only for checking them",
        "clips": clips,
    }
    ground_truth_path = output_root / "ground_truth.json"
    ground_truth_path.write_text(json.dumps(ground_truth, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(ground_truth_path)

    payload_files = sorted(path for path in written if path.is_file())
    manifest = {
        "schema_version": 1,
        "package_version": FALLBACK_VERSION,
        "generator": {
            "path": "assignment3/data/generate_fallback.py",
            "sha256": sha256(Path(__file__).resolve()),
            "ffmpeg_version": encoder_version,
        },
        "license": "Course Educational Asset License 1.0 (see LICENSE.md)",
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in payload_files
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(manifest_path)
    return written


def verify_fallback_package(package_root: Path = DEFAULT_FALLBACK_ROOT) -> dict[str, Any]:
    """Verify every manifest size/hash and return parsed metadata."""

    package_root = Path(package_root)
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        path = package_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"fallback size mismatch: {filename}")
        actual = sha256(path)
        if actual != expected["sha256"]:
            raise RuntimeError(f"fallback SHA-256 mismatch: {filename}")
    return manifest


if __name__ == "__main__":
    paths = generate_fallback_package()
    verify_fallback_package()
    print(f"generated and verified {len(paths)} fallback files under {DEFAULT_FALLBACK_ROOT}")
