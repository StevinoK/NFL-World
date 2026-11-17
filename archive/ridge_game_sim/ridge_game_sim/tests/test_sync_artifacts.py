from contextlib import contextmanager

import pytest

from ridge_game_sim.sync_artifacts import upload_missing_bundle_files


class Volume:
    def __init__(self, files):
        self.files = dict(files)
        self.uploads = []

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        content = self.files[path]
        yield content[:2]
        yield content[2:]

    @contextmanager
    def batch_upload(self, force=False):
        assert force is False
        yield self

    def put_file(self, local, remote):
        assert remote not in self.files
        self.files[remote] = local.read_bytes()
        self.uploads.append(remote)


def test_upload_retries_skip_matching_files_and_complete_partial_bundle(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"weights")
    inputs = tmp_path / "weekly_inputs"
    inputs.mkdir()
    (inputs / "input_manifest.json").write_bytes(b"manifest")
    destination = "/bundles/hash/runtime"
    volume = Volume({f"{destination}/weekly_inputs/input_manifest.json": b"manifest"})
    upload_missing_bundle_files(volume, tmp_path, destination)
    assert volume.uploads == [f"{destination}/model.pt"]
    upload_missing_bundle_files(volume, tmp_path, destination)
    assert volume.uploads == [f"{destination}/model.pt"]


def test_upload_refuses_conflicting_remote_files_before_writing(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"missing")
    (tmp_path / "b.pt").write_bytes(b"correct")
    volume = Volume({"/runtime/b.pt": b"wrong"})
    with pytest.raises(ValueError, match="Immutable remote bundle file differs"):
        upload_missing_bundle_files(volume, tmp_path, "/runtime")
    assert volume.uploads == []