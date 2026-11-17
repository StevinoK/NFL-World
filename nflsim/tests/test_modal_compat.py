def test_deployment_packages_compatibility_import():
    import ast
    from nflsim.paths import ROOT
    tree = ast.parse((ROOT / "remote/modal_app.py").read_text())
    packaged = {
        value.value
        for node in ast.walk(tree) if isinstance(node, ast.Tuple)
        for value in node.elts if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }
    assert "remote/modal_compat.py" in packaged
    assert all((ROOT / name).is_file() for name in packaged if name.endswith(".py"))

from unittest.mock import Mock

import pytest

from nflsim.remote.modal_compat import artifact_mount


def test_readonly_options_when_available():
    volume = Mock()
    assert artifact_mount(volume) is volume.with_mount_options.return_value
    volume.with_mount_options.assert_called_once_with(read_only=True)


def test_old_sdk_mount_is_explicitly_warned():
    volume = object()
    with pytest.warns(RuntimeWarning, match="mount-level"):
        assert artifact_mount(volume) is volume