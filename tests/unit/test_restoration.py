from utils import restoration


def test_restore_document_image_opencv_backend_preserves_contract(blurred_image):
    result = restoration.restore_document_image(
        blurred_image,
        backend="opencv",
        mode="ocr",
        use_deconvolution=False,
    )

    assert result.backend == restoration.OPENCV_RESTORATION
    assert result.image.shape == blurred_image.shape
    assert result.image.dtype == blurred_image.dtype
    assert result.used_fallback is False


def test_docres_backend_without_command_falls_back_to_opencv(blurred_image, monkeypatch):
    monkeypatch.delenv("DOCRES_COMMAND", raising=False)
    monkeypatch.setattr(restoration, "DEFAULT_DOCRES_SCRIPT", restoration.PROJECT_ROOT / "missing_run_docres.py")

    result = restoration.restore_document_image(
        blurred_image,
        backend="docres",
        mode="natural",
        use_deconvolution=False,
    )

    assert result.backend == restoration.OPENCV_RESTORATION
    assert result.image.shape == blurred_image.shape
    assert result.used_fallback is True
    assert "DOCRES_COMMAND" in result.error

def test_docres_configuration_uses_local_wrapper_when_present(tmp_path, monkeypatch):
    script = tmp_path / "run_docres.py"
    docres_weight = tmp_path / "docres.pkl"
    mbd_weight = tmp_path / "mbd.pkl"
    script.write_text("print('stub')", encoding="utf-8")
    docres_weight.write_text("stub", encoding="utf-8")
    mbd_weight.write_text("stub", encoding="utf-8")

    monkeypatch.delenv("DOCRES_COMMAND", raising=False)
    monkeypatch.setattr(restoration, "DEFAULT_DOCRES_SCRIPT", script)
    monkeypatch.setattr(restoration, "DEFAULT_DOCRES_WEIGHTS", (docres_weight, mbd_weight))

    command = restoration.get_docres_command_template()

    assert restoration.is_docres_configured()
    assert str(script) in command
    assert "--task {task}" in command
