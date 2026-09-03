"""Generate deterministic PA3 data and download verified VGG planar sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assignment3.common.pa3_utils import make_homography_case, make_translation_pair  # noqa: E402
from assignment3.data.generate_fallback import (  # noqa: E402
    DEFAULT_FALLBACK_ROOT,
    generate_fallback_package,
    verify_fallback_package,
)


ASSETS = {
    "wall": {
        "url": "https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/wall.tar.gz",
        "sha256": "3cb3df75f40471f093adf6647d13df6a9f4312f7a222fc591d03f6c08e086344",
    },
    "graf": {
        "url": "https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz",
        "sha256": "999871b945ee968a00a0d5f9af957d1382fb9dae1511cdee9553366817b53b5b",
    },
}

REQUIRED_B4_ASSET = "wall"


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_synthetic(output_root: Path) -> list[Path]:
    """Generate every released deterministic synthetic case."""

    synthetic_root = output_root / "synthetic"
    synthetic_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fraction in (0.0, 0.2, 0.4, 0.6):
        for seed in range(16720, 16725):
            case = make_homography_case(seed=seed, outlier_fraction=fraction)
            destination = synthetic_root / f"homography_outliers_{int(100*fraction):02d}_seed_{seed}.npz"
            np.savez_compressed(
                destination,
                src_xy=case["src_xy"],
                dst_xy=case["dst_xy"],
                dst_clean_xy=case["dst_clean_xy"],
                true_inliers=case["true_inliers"],
                H_true=case["H_true"],
                image_size=np.asarray(case["image_size"], dtype=np.int64),
            )
            written.append(destination)

    translation_specs = {
        "zero": ((0.0, 0.0), 0.0, 16720),
        "one_pixel": ((1.0, 0.0), 0.0, 16721),
        "large": ((8.0, -5.0), 0.0, 16722),
        "subpixel": ((3.25, -2.5), 0.0, 16723),
        "brightness_violation": ((2.0, 1.0), 0.12, 16724),
    }
    for name, (shift, beta, seed) in translation_specs.items():
        template, image, true_p = make_translation_pair(shift, beta=beta, seed=seed)
        destination = synthetic_root / f"translation_{name}.npz"
        np.savez_compressed(destination, template=template, image=image, true_p=true_p, beta=np.array(beta))
        written.append(destination)

    manifest = {
        "generator": "assignment3/data/setup_data.py",
        "shared_seed": 16720,
        "files": [str(path.relative_to(output_root)) for path in written],
    }
    manifest_path = output_root / "synthetic_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    written.append(manifest_path)
    return written


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            resolved_member = (destination / member.name).resolve()
            if resolved_destination not in resolved_member.parents and resolved_member != resolved_destination:
                raise RuntimeError(f"archive contains unsafe path: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"archive contains unsupported member type: {member.name}")
        # Copy validated regular files/directories ourselves.  This is safe,
        # warning-free, and compatible with Python 3.10; tarfile's modern
        # filter="data" interface is newer.
        for member in tar.getmembers():
            member_path = destination / member.name
            if member.isdir():
                member_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                member_path.parent.mkdir(parents=True, exist_ok=True)
                if member_path.exists():
                    # Some historical archives store executable/read-only
                    # modes.  Make a prior generated copy replaceable so the
                    # setup command remains idempotent.
                    member_path.chmod(0o644)
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"could not read archive member: {member.name}")
                with source, member_path.open("wb") as output:
                    shutil.copyfileobj(source, output)
            else:
                raise RuntimeError(f"archive contains unsupported member type: {member.name}")


def download_external(output_root: Path, asset_names: tuple[str, ...] | None = None) -> list[Path]:
    """Download, verify, and extract selected Oxford VGG sequences.

    ``asset_names=None`` retains the original API behavior and downloads every
    configured sequence (required ``wall`` plus optional/debug ``graf``).
    """

    archive_root = output_root / "downloads"
    sequence_root = output_root / "vgg_affine"
    archive_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    names = tuple(ASSETS) if asset_names is None else tuple(asset_names)
    if not names:
        return written
    unknown = sorted(set(names) - set(ASSETS))
    if unknown:
        raise ValueError(f"unknown Oxford VGG asset(s): {', '.join(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError("asset_names must not contain duplicates")
    for name in names:
        metadata = ASSETS[name]
        archive_path = archive_root / f"{name}.tar.gz"
        if not archive_path.exists() or sha256(archive_path) != metadata["sha256"]:
            print(f"Downloading {name} from {metadata['url']}")
            urllib.request.urlretrieve(metadata["url"], archive_path)
        actual = sha256(archive_path)
        if actual != metadata["sha256"]:
            raise RuntimeError(f"checksum mismatch for {name}: expected {metadata['sha256']}, received {actual}")
        destination = sequence_root / name
        _safe_extract(archive_path, destination)
        written.extend(sorted(destination.iterdir()))
    return written


def download_b4(output_root: Path) -> list[Path]:
    """Download the checksum-verified ``wall`` sequence required by graded B4."""

    return download_external(output_root, (REQUIRED_B4_ASSET,))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="generate deterministic synthetic cases")
    parser.add_argument("--b4", action="store_true", help="download required checksum-verified Oxford wall data")
    parser.add_argument(
        "--external",
        action="store_true",
        help="download all Oxford data (required wall plus optional/debug graf; legacy-compatible)",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="regenerate the release-included, staff-approval-only C/D fallback package",
    )
    parser.add_argument(
        "--fallback-output",
        type=Path,
        default=DEFAULT_FALLBACK_ROOT,
        help="fallback package destination (default: assignment3/data/fallback)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run synthetic setup, all Oxford downloads, and fallback regeneration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
        help="output directory (default: assignment3/data/generated)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    any_action = args.synthetic or args.b4 or args.external or args.fallback or args.all
    do_synthetic = args.synthetic or args.all or not any_action
    do_external = args.external or args.all
    do_b4 = args.b4 and not do_external
    args.output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if do_synthetic:
        written.extend(generate_synthetic(args.output))
    if do_external:
        written.extend(download_external(args.output))
    elif do_b4:
        written.extend(download_b4(args.output))
    if args.fallback or args.all:
        written.extend(generate_fallback_package(args.fallback_output))
        verify_fallback_package(args.fallback_output)
    print(f"PA3 data setup complete: {len(written)} files prepared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
