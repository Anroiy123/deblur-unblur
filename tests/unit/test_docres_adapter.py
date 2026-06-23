import os

from scripts.run_docres import build_docres_environment


def test_docres_environment_injects_numpy_bool_compatibility(tmp_path):
    environment = build_docres_environment(tmp_path)

    assert (tmp_path / "sitecustomize.py").exists()
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path)
    shim = (tmp_path / "sitecustomize.py").read_text(encoding="utf-8")
    assert "np.bool = bool" in shim
