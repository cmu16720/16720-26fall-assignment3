"""Generate deterministic PA3 data and download verified VGG planar sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import urllib.request

import numpy as np


# Works from the staff tree (assignment3/data/...) and from the flat student
# repository (data/...), where there is no outer package directory.
ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = ASSIGNMENT_ROOT.parent if ASSIGNMENT_ROOT.name.startswith("assignment") else ASSIGNMENT_ROOT
for _candidate in (ROOT, ASSIGNMENT_ROOT):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

try:  # staff tree
    from assignment3.common.pa3_utils import make_homography_case, make_translation_pair  # noqa: E402
    from assignment3.data.generate_fallback import (  # noqa: E402
        DEFAULT_FALLBACK_ROOT,
        generate_fallback_package,
        verify_fallback_package,
    )
except ModuleNotFoundError:  # flat student repository
    from common.pa3_utils import make_homography_case, make_translation_pair  # noqa: E402
    from data.generate_fallback import (  # noqa: E402
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

# Pinned backup of the original Oxford files. Every file is checked before use.
WALL_MIRROR = (
    "https://raw.githubusercontent.com/AnoopP7/CSS587GaLAM/"
    "1dc33faeb704ee614abda8645fa217959678d82f/data/wall/"
)
WALL_FILE_SHA256 = {
    "H1to2p": "5106d4e1e974c1cfdb2cb362a68b35fc810e0878341ec0bb8bfa16093221b0a0",
    "H1to3p": "97d5e44fe81e8310693f55ca1a17bee766401a5581d366716111f157606cabe9",
    "H1to4p": "cd4bbb982060e5142d90d4da431c27104aaf6a923260be0d6e8bb2970a540095",
    "H1to5p": "d547753f4e5b904faf25da29ff3f20ed19dc4498b6a3065f9ee5cce7c88eb9ea",
    "H1to6p": "13e7c93bdc31fcc0f56bb819e0b51deb109e81cf8a9d7adbc269b66587e8abef",
    "img1.ppm": "96140c5a649f6cfaf002fc06fdd97f22e7eec322d19aa522d9f198e90258419c",
    "img2.ppm": "e7a4a1ea610ee1e4c8c21654c3200764f4a405b92b814d230cf219fb2ae27b6a",
    "img3.ppm": "7f21d250f008e5cbd55cca6fc342dfd175c730728ccc39280cc0ed91a05738c1",
    "img4.ppm": "a1fdfdbf4daa8afd6dc6fd1a0ba41151f9a9f1ca6475ed29864cc4099296c2d1",
    "img5.ppm": "4114ddb2fd6dcaf47047b437c10a0a8d9280f6a014d5268907b0776e6cb4745e",
    "img6.ppm": "5ff1c5e013cc6f5cf2936b9a1e0eef2b01155356b9346bea9faa47edbe98363e"
}
DOWNLOAD_TIMEOUT = 15



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
        "brightness_violation": ((2.0, 1.0), 0.2, 16724),
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
        # Copy validated regular files and directories ourselves. This is safe,
        # prints no warnings, and works on Python 3.10, which predates tarfile's
        # filter="data" option.
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


def _download_verified(url: str, destination: Path, expected: str) -> None:
    """Only replace a cached file after a complete, checksum-valid download."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        if sha256(temporary) != expected:
            raise OSError(f"checksum mismatch for {destination.name} from {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _wall_files_valid(destination: Path) -> bool:
    return all(
        (destination / name).is_file() and sha256(destination / name) == expected
        for name, expected in WALL_FILE_SHA256.items()
    )


def _download_wall_mirror(destination: Path) -> list[Path]:
    for name, expected in WALL_FILE_SHA256.items():
        path = destination / name
        if not path.is_file() or sha256(path) != expected:
            _download_verified(WALL_MIRROR + name, path, expected)
    return sorted(destination / name for name in WALL_FILE_SHA256)


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
        destination = sequence_root / name
        if name == "wall" and _wall_files_valid(destination):
            written.extend(sorted(destination / file for file in WALL_FILE_SHA256))
            continue
        if not archive_path.exists() or sha256(archive_path) != metadata["sha256"]:
            print(f"Downloading {name} from {metadata['url']}")
            try:
                _download_verified(metadata["url"], archive_path, metadata["sha256"])
            except OSError as error:
                if name != "wall":
                    raise
                print(f"Oxford download unavailable ({error}); trying verified wall backup.")
                written.extend(_download_wall_mirror(destination))
                continue
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
        help="download all Oxford data (required wall plus optional graf for debugging)",
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
