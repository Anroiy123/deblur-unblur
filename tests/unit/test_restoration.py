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
    assert result.requested_backend == restoration.OPENCV_RESTORATION
    assert result.attempted_backends == (restoration.OPENCV_RESTORATION,)


def test_docres_backend_without_command_falls_back_through_chain_to_opencv(blurred_image, monkeypatch):
    monkeypatch.delenv("DOCRES_COMMAND", raising=False)
    monkeypatch.delenv("RESTORMER_COMMAND", raising=False)
    monkeypatch.delenv("RESTORMER_REPO_DIR", raising=False)
    monkeypatch.delenv("NAFNET_COMMAND", raising=False)
    monkeypatch.delenv("NAFNET_REPO_DIR", raising=False)
    monkeypatch.delenv("NAFNET_OPTIONS", raising=False)
    monkeypatch.setattr(restoration, "DEFAULT_DOCRES_SCRIPT", restoration.PROJECT_ROOT / "missing_run_docres.py")
    monkeypatch.setattr(restoration, "DEFAULT_RESTORMER_SCRIPT", restoration.PROJECT_ROOT / "missing_run_restormer.py")
    monkeypatch.setattr(restoration, "DEFAULT_NAFNET_SCRIPT", restoration.PROJECT_ROOT / "missing_run_nafnet.py")

    result = restoration.restore_document_image(
        blurred_image,
        backend="docres",
        mode="natural",
        use_deconvolution=False,
    )

    assert result.backend == restoration.OPENCV_RESTORATION
    assert result.image.shape == blurred_image.shape
    assert result.used_fallback is True
    assert result.requested_backend == restoration.DOCRES_RESTORATION
    assert result.attempted_backends == (
        restoration.DOCRES_RESTORATION,
        restoration.RESTORMER_RESTORATION,
        restoration.NAFNET_RESTORATION,
        restoration.OPENCV_RESTORATION,
    )
    assert "docres:" in result.error.lower()
    assert "restormer:" in result.error.lower()
    assert "nafnet:" in result.error.lower()


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


def test_restormer_configuration_uses_local_wrapper_when_repo_present(tmp_path, monkeypatch):
    script = tmp_path / "run_restormer.py"
    repo_dir = tmp_path / "Restormer"
    demo_script = repo_dir / "demo.py"
    repo_dir.mkdir()
    demo_script.write_text("print('stub')", encoding="utf-8")
    script.write_text("print('stub')", encoding="utf-8")

    monkeypatch.delenv("RESTORMER_COMMAND", raising=False)
    monkeypatch.setenv("RESTORMER_REPO_DIR", str(repo_dir))
    monkeypatch.setattr(restoration, "DEFAULT_RESTORMER_SCRIPT", script)

    command = restoration.get_restormer_command_template()

    assert str(script) in command
    assert str(repo_dir) in command
    assert "--task {task}" in command


def test_nafnet_configuration_uses_local_wrapper_when_repo_and_options_present(tmp_path, monkeypatch):
    script = tmp_path / "run_nafnet.py"
    repo_dir = tmp_path / "NAFNet"
    demo_script = repo_dir / "basicsr" / "demo.py"
    options_path = repo_dir / "options" / "test" / "REDS" / "NAFNet-width64.yml"
    demo_script.parent.mkdir(parents=True)
    options_path.parent.mkdir(parents=True)
    demo_script.write_text("print('stub')", encoding="utf-8")
    options_path.write_text("name: test", encoding="utf-8")
    script.write_text("print('stub')", encoding="utf-8")

    monkeypatch.delenv("NAFNET_COMMAND", raising=False)
    monkeypatch.setenv("NAFNET_REPO_DIR", str(repo_dir))
    monkeypatch.setenv("NAFNET_OPTIONS", str(options_path))
    monkeypatch.setattr(restoration, "DEFAULT_NAFNET_SCRIPT", script)

    command = restoration.get_nafnet_command_template()

    assert str(script) in command
    assert str(repo_dir) in command
    assert str(options_path) in command
    assert "--task {task}" in command


def test_auto_backend_promotes_restormer_when_docres_fails(blurred_image, monkeypatch):
    def fake_docres(_image, task="deblurring", timeout_seconds=180):
        raise RuntimeError(f"DocRes unavailable for {task}")

    def fake_restormer(_image, task="deblurring", timeout_seconds=180):
        return restoration.RestorationResult(
            image=blurred_image.copy(),
            backend=restoration.RESTORMER_RESTORATION,
            message=f"Restormer restored with {task}",
        )

    monkeypatch.setattr(restoration, "restore_with_docres", fake_docres)
    monkeypatch.setattr(restoration, "restore_with_restormer", fake_restormer)

    result = restoration.restore_document_image(
        blurred_image,
        backend="auto",
        mode="ocr",
        restoration_task="deblurring",
    )

    assert result.backend == restoration.RESTORMER_RESTORATION
    assert result.used_fallback is True
    assert result.requested_backend == restoration.AUTO_RESTORATION
    assert result.attempted_backends == (
        restoration.DOCRES_RESTORATION,
        restoration.RESTORMER_RESTORATION,
    )
    assert "docres:" in result.error.lower()


def test_restore_document_image_unexpected_failure_falls_back_to_opencv(blurred_image, monkeypatch):
    def broken_chain(_backend):
        raise RuntimeError("planner crashed")

    monkeypatch.setattr(restoration, "get_restoration_fallback_chain", broken_chain)

    result = restoration.restore_document_image(
        blurred_image,
        backend="auto",
        mode="natural",
        use_deconvolution=False,
        fallback_to_opencv=True,
    )

    assert result.backend == restoration.OPENCV_RESTORATION
    assert result.used_fallback is True
    assert result.requested_backend == restoration.AUTO_RESTORATION
    assert result.attempted_backends == (restoration.OPENCV_RESTORATION,)
    assert "unexpected:" in result.error.lower()
