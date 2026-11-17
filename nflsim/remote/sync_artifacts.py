from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nflsim.bundle import validate_bundle, verify_bundle_hash
from nflsim.paths import APP_NAME, ROOT, VOLUME_NAME, check_remote_names, digest, inside, write_json


def upload_missing_bundle_files(volume, directory: Path, destination: str) -> None:
    missing = []
    for local in sorted(directory.rglob("*")):
        if not local.is_file():
            continue
        remote = f"{destination}/{local.relative_to(directory).as_posix()}"
        checksum = hashlib.sha256()
        try:
            for chunk in volume.read_file(remote):
                checksum.update(chunk)
        except FileNotFoundError:
            missing.append((local, remote))
            continue
        if checksum.hexdigest() != digest(local):
            raise ValueError(f"Immutable remote bundle file differs: {remote}; upload refused")
    if missing:
        with volume.batch_upload(force=False) as batch:
            for local, remote in missing:
                batch.put_file(local, remote)


def upload_bundle(artifact_dir: Path, volume_name: str = VOLUME_NAME, environment: str | None = None,
                 *, confirmed: bool = False) -> str:
    if not confirmed:
        raise ValueError("Explicit upload confirmation is required")
    check_remote_names(APP_NAME, volume_name)
    directory = inside(artifact_dir)
    validate_bundle(directory)
    bundle_hash = verify_bundle_hash(directory)
    for lock in (ROOT / "benchmarks").rglob("active_bundle.json"):
        if json.loads(lock.read_text()).get("bundle_hash") == bundle_hash:
            raise ValueError("Bundle is in use by a benchmark; upload refused")
    import modal
    volume = modal.Volume.from_name(volume_name, environment_name=environment, create_if_missing=True)
    destination = f"/bundles/{bundle_hash}/runtime"
    upload_missing_bundle_files(volume, directory, destination)
    write_json(ROOT / "config/last_upload.json", {"bundle_hash": bundle_hash, "volume": volume_name,
        "environment": environment, "path": destination, "redeploy_required": False})
    return bundle_hash


def main():
    parser = argparse.ArgumentParser(description="Explicit upload to immutable ridge-only Modal volume path")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/runtime")
    parser.add_argument("--volume", default=VOLUME_NAME)
    parser.add_argument("--environment")
    parser.add_argument("--confirm-upload", action="store_true", required=True)
    args = parser.parse_args()
    bundle_hash = upload_bundle(args.artifact_dir, args.volume, args.environment, confirmed=args.confirm_upload)
    print(f"Uploaded immutable bundle {bundle_hash}. The existing Ridge Modal app reads hash-addressed bundles without redeploying.")


if __name__ == "__main__":
    main()