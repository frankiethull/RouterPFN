"""Publish helper: export a versioned release bundle (MIT).

No model weights are published — the artifacts are the pip adapter
source (this repo) and the synthetic routing corpus. Produces a
directory (or tarball) with corpus + report + manifest for attaching
to a GitHub release. File-based on purpose: no registry accounts,
no tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a RouterPFN release bundle")
    ap.add_argument("--corpus", type=Path, default=Path("data/raw/teacher-v1.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("reports/bench.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("dist"))
    ap.add_argument("--version", default="v0")
    ap.add_argument("--tarball", action="store_true")
    args = ap.parse_args()

    bundle = args.out_dir / f"RouterPFN-{args.version}"
    bundle.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "package": "RouterPFN",
        "version": args.version,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {},
    }
    for src in (args.corpus, args.report):
        if src.exists():
            dest = bundle / src.name
            dest.write_bytes(src.read_bytes())
            manifest["files"][src.name] = {
                "sha256": sha256_of(dest),
                "bytes": dest.stat().st_size,
            }
        else:
            print(f"skip missing {src}")
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"bundle: {bundle}")
    if args.tarball:
        tgz = args.out_dir / f"RouterPFN-{args.version}.tar.gz"
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(bundle, arcname=bundle.name)
        print(f"tarball: {tgz}")


if __name__ == "__main__":
    main()
