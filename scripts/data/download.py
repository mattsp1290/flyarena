"""Idempotent, hash-verified downloader for the pinned MaleCNS v1.0 source
files this compiler consumes.

Source: the Male CNS connectome, Janelia FlyEM Project (HHMI), in
collaboration with the MRC Laboratory of Molecular Biology (Cambridge) and
Google Research. Public bulk data is served unauthenticated from the Google
Cloud Storage bucket `gs://flyem-male-cns` (also readable over HTTPS via
`storage.googleapis.com`); no neuPrint auth token is required for these flat
connectome export files. See docs/data-provenance.md for the full citation
and confirmation of how this was verified.

Dataset: "male-cns:v1.0" (release `v1.0`, filenames sometimes still carry the
export tag `minconf-0.5`, the presynapse confidence threshold used when the
flat tables were generated).
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), per
https://male-cns.janelia.org/download.

Every entry below is pinned to an exact URL and a sha256 recorded at
download time by this script's author; `download.py` re-verifies the hash
on every run (including for files already present) and refuses to proceed
on a mismatch rather than silently using stale or tampered data. Raw files
are never committed (`data/raw/` is gitignored) -- only the compiled,
already-small artifact under `public/data/` is.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"

_BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"


@dataclass(frozen=True)
class SourceFile:
    """One pinned MaleCNS v1.0 source file."""

    filename: str
    url: str
    sha256: str
    size_bytes: int
    description: str


# Pinned exact sources. Only the three flat-connectome tables this compiler
# actually needs are listed -- the bucket also hosts multi-gigabyte
# synapse-point-cloud and neurotransmitter-per-synapse tables
# (`syn-points-*`, `syn-partners-*`, `tbar-neurotransmitters-*`) that are not
# required to build the node/edge subgraph and were intentionally not
# downloaded (see docs/data-provenance.md's "What was not downloaded"
# section).
SOURCE_FILES: tuple[SourceFile, ...] = (
    SourceFile(
        filename="body-annotations-male-cns-v1.0-minconf-0.5.feather",
        url=f"{_BASE_URL}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        sha256="2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
        size_bytes=14_483_314,
        description=(
            "Curated per-neuron annotations (superclass/class/status/type/side/"
            "nerve), excluding neurotransmitter properties."
        ),
    ),
    SourceFile(
        filename="body-neurotransmitters-male-cns-v1.0.feather",
        url=f"{_BASE_URL}/body-neurotransmitters-male-cns-v1.0.feather",
        sha256="95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
        size_bytes=43_282_834,
        description="Aggregate per-neuron neurotransmitter predictions (consensus_nt, predicted_nt, ...).",
    ),
    SourceFile(
        filename="connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        url=f"{_BASE_URL}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        sha256="e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
        size_bytes=1_051_241_946,
        description=(
            "Full segment-to-segment connection graph: (body_pre, body_post, weight) "
            "with weight = aggregated synapse count between that ordered pair."
        ),
    ),
)


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_downloaded(source: SourceFile, dest_dir: Path = RAW_DATA_DIR) -> Path:
    """Download `source` into `dest_dir` if not already present with a
    matching hash. Idempotent: a second call with the file already present
    and correct only re-hashes (no network request). Raises on any hash
    mismatch, whether pre-existing or freshly downloaded, instead of
    silently proceeding."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.filename

    if dest.exists():
        actual = _sha256_of_file(dest)
        if actual == source.sha256:
            print(f"[ok]      {source.filename} already present, sha256 verified")
            return dest
        raise RuntimeError(
            f"{dest} exists but sha256 {actual} does not match pinned {source.sha256}; "
            "refusing to use a stale/corrupt file. Delete it and re-run to re-download."
        )

    print(f"[fetch]   {source.filename} <- {source.url}")
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(source.url, timeout=120) as response, open(tmp_dest, "wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)

    actual = _sha256_of_file(tmp_dest)
    if actual != source.sha256:
        tmp_dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded {source.filename} sha256 {actual} does not match pinned {source.sha256}"
        )
    tmp_dest.rename(dest)
    print(f"[done]    {source.filename} sha256 verified")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=RAW_DATA_DIR,
        help="directory to download into (default: data/raw/, gitignored)",
    )
    args = parser.parse_args(argv)

    for source in SOURCE_FILES:
        ensure_downloaded(source, dest_dir=args.dest)

    print(f"\nAll {len(SOURCE_FILES)} pinned source files present and hash-verified in {args.dest}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
