from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2

from utils.enhancement import enhance_image

AUTO_RESTORATION = "auto"
OPENCV_RESTORATION = "opencv"
DOCRES_RESTORATION = "docres"
RESTORMER_RESTORATION = "restormer"
NAFNET_RESTORATION = "nafnet"
SUPPORTED_RESTORATION_BACKENDS = (
    AUTO_RESTORATION,
    OPENCV_RESTORATION,
    DOCRES_RESTORATION,
    RESTORMER_RESTORATION,
    NAFNET_RESTORATION,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCRES_SCRIPT = PROJECT_ROOT / "scripts" / "run_docres.py"
DEFAULT_RESTORMER_SCRIPT = PROJECT_ROOT / "scripts" / "run_restormer.py"
DEFAULT_NAFNET_SCRIPT = PROJECT_ROOT / "scripts" / "run_nafnet.py"
DEFAULT_AI_VENV = PROJECT_ROOT / ".venv-ai312" / "Scripts" / "python.exe"
DEFAULT_ALT_VENV = PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe"
DEFAULT_DOCRES_WEIGHTS = (
    PROJECT_ROOT / "external" / "DocRes" / "checkpoints" / "docres.pkl",
    PROJECT_ROOT / "external" / "DocRes" / "data" / "MBD" / "checkpoint" / "mbd.pkl",
)
DEFAULT_RESTORMER_DIR = PROJECT_ROOT / "external" / "Restormer"
DEFAULT_NAFNET_DIR = PROJECT_ROOT / "external" / "NAFNet"
DEFAULT_NAFNET_OPTIONS = (
    DEFAULT_NAFNET_DIR / "options" / "test" / "GoPro" / "NAFNet-width64.yml"
)
RESTORATION_BACKEND_LABELS = {
    AUTO_RESTORATION: "Tự động tốt nhất",
    OPENCV_RESTORATION: "OpenCV",
    DOCRES_RESTORATION: "DocRes",
    RESTORMER_RESTORATION: "Restormer",
    NAFNET_RESTORATION: "NAFNet",
}


@dataclass(frozen=True)
class RestorationResult:
    image: object
    backend: str
    message: str = ""
    error: str = ""
    used_fallback: bool = False
    requested_backend: str = ""
    attempted_backends: tuple[str, ...] = ()

    @property
    def ok(self):
        return not self.error


def normalize_restoration_backend(backend):
    if not backend:
        return AUTO_RESTORATION

    normalized = str(backend).strip().lower()
    if normalized in ("auto", "best", "recommended", "smart"):
        return AUTO_RESTORATION
    if normalized in ("opencv", "classical", "baseline"):
        return OPENCV_RESTORATION
    if normalized in ("docres", "ai", "document_ai"):
        return DOCRES_RESTORATION
    if normalized in ("restormer", "restoration_transformer"):
        return RESTORMER_RESTORATION
    if normalized in ("nafnet", "naf"):
        return NAFNET_RESTORATION

    raise ValueError(f"Unsupported restoration backend: {backend}")


def get_restoration_backend_label(backend):
    normalized = normalize_restoration_backend(backend)
    return RESTORATION_BACKEND_LABELS[normalized]


def is_docres_configured():
    return bool(get_docres_command_template())


def is_backend_configured(backend):
    backend = normalize_restoration_backend(backend)
    if backend == OPENCV_RESTORATION:
        return True
    if backend == DOCRES_RESTORATION:
        return bool(get_docres_command_template())
    if backend == RESTORMER_RESTORATION:
        return bool(get_restormer_command_template())
    if backend == NAFNET_RESTORATION:
        return bool(get_nafnet_command_template())
    if backend == AUTO_RESTORATION:
        return any(
            is_backend_configured(candidate)
            for candidate in get_restoration_fallback_chain(AUTO_RESTORATION)
            if candidate != AUTO_RESTORATION
        )
    return False


def get_docres_command_template():
    explicit_command = os.environ.get("DOCRES_COMMAND")
    if explicit_command:
        return explicit_command

    python_executable = get_backend_python_executable(DOCRES_RESTORATION)
    if DEFAULT_DOCRES_SCRIPT.exists() and all(weight.exists() for weight in DEFAULT_DOCRES_WEIGHTS):
        return (
            f'"{python_executable}" "{DEFAULT_DOCRES_SCRIPT}" '
            '--task {task} --input "{input}" --output "{output}"'
        )

    return ""


def get_backend_python_executable(backend):
    backend = normalize_restoration_backend(backend)

    specific_keys = {
        DOCRES_RESTORATION: "DOCRES_PYTHON",
        RESTORMER_RESTORATION: "RESTORMER_PYTHON",
        NAFNET_RESTORATION: "NAFNET_PYTHON",
    }
    env_keys = [specific_keys.get(backend), "RESTORATION_PYTHON"]

    for key in env_keys:
        if not key:
            continue
        value = os.environ.get(key)
        if value and Path(value).exists():
            return str(Path(value))

    for candidate in (DEFAULT_AI_VENV, DEFAULT_ALT_VENV):
        if candidate.exists():
            return str(candidate)

    return sys.executable


def get_restormer_command_template():
    explicit_command = os.environ.get("RESTORMER_COMMAND")
    if explicit_command:
        return explicit_command

    repo_dir = Path(os.environ.get("RESTORMER_REPO_DIR", DEFAULT_RESTORMER_DIR))
    python_executable = get_backend_python_executable(RESTORMER_RESTORATION)
    if DEFAULT_RESTORMER_SCRIPT.exists() and (repo_dir / "demo.py").exists():
        return (
            f'"{python_executable}" "{DEFAULT_RESTORMER_SCRIPT}" '
            '--task {task} --input "{input}" --output "{output}" '
            f'--repo-dir "{repo_dir}"'
        )

    return ""


def get_nafnet_command_template():
    explicit_command = os.environ.get("NAFNET_COMMAND")
    if explicit_command:
        return explicit_command

    repo_dir = Path(os.environ.get("NAFNET_REPO_DIR", DEFAULT_NAFNET_DIR))
    options_path = Path(os.environ.get("NAFNET_OPTIONS", DEFAULT_NAFNET_OPTIONS))
    python_executable = get_backend_python_executable(NAFNET_RESTORATION)
    if (
        DEFAULT_NAFNET_SCRIPT.exists()
        and (repo_dir / "basicsr" / "demo.py").exists()
        and options_path.exists()
    ):
        return (
            f'"{python_executable}" "{DEFAULT_NAFNET_SCRIPT}" '
            '--task {task} --input "{input}" --output "{output}" '
            f'--repo-dir "{repo_dir}" --opt "{options_path}"'
        )

    return ""


def restore_with_opencv(image, mode="ocr", use_deconvolution=False):
    restored = enhance_image(image, use_deconvolution=use_deconvolution, mode=mode)
    return RestorationResult(
        image=restored,
        backend=OPENCV_RESTORATION,
        message="OpenCV baseline restoration completed",
        requested_backend=OPENCV_RESTORATION,
        attempted_backends=(OPENCV_RESTORATION,),
    )


def _run_command_backend(backend, image, task="deblurring", timeout_seconds=180):
    if backend == DOCRES_RESTORATION:
        command_template = get_docres_command_template()
    elif backend == RESTORMER_RESTORATION:
        command_template = get_restormer_command_template()
    elif backend == NAFNET_RESTORATION:
        command_template = get_nafnet_command_template()
    else:
        raise ValueError(f"Unsupported command restoration backend: {backend}")

    if not command_template:
        raise RuntimeError(
            f"{get_restoration_backend_label(backend)} is not configured. "
            f"Set a command template for backend={backend} or install its local adapter prerequisites."
        )

    with tempfile.TemporaryDirectory(prefix=f"deblur_{backend}_") as temp_dir:
        input_path = os.path.join(temp_dir, "input.png")
        output_path = os.path.join(temp_dir, "output.png")

        if not cv2.imwrite(input_path, image):
            raise RuntimeError(f"Unable to write temporary {backend} input image")

        command = command_template.format(input=input_path, output=output_path, task=task)
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"{get_restoration_backend_label(backend)} command failed with exit code "
                f"{completed.returncode}: {stderr}"
            )

        restored = cv2.imread(output_path, cv2.IMREAD_COLOR)
        if restored is None:
            raise RuntimeError(
                f"{get_restoration_backend_label(backend)} completed but did not produce a "
                "readable output image"
            )

        return RestorationResult(
            image=restored,
            backend=backend,
            message=f"{get_restoration_backend_label(backend)} restoration completed with task={task}",
            requested_backend=backend,
            attempted_backends=(backend,),
        )


def restore_with_docres(image, task="end2end", timeout_seconds=180):
    return _run_command_backend(
        DOCRES_RESTORATION,
        image,
        task=task,
        timeout_seconds=timeout_seconds,
    )


def restore_with_restormer(image, task="deblurring", timeout_seconds=180):
    return _run_command_backend(
        RESTORMER_RESTORATION,
        image,
        task=task,
        timeout_seconds=timeout_seconds,
    )


def restore_with_nafnet(image, task="deblurring", timeout_seconds=180):
    return _run_command_backend(
        NAFNET_RESTORATION,
        image,
        task=task,
        timeout_seconds=timeout_seconds,
    )


def get_restoration_fallback_chain(backend):
    backend = normalize_restoration_backend(backend)
    if backend in (AUTO_RESTORATION, DOCRES_RESTORATION):
        return (
            DOCRES_RESTORATION,
            RESTORMER_RESTORATION,
            NAFNET_RESTORATION,
            OPENCV_RESTORATION,
        )
    if backend == RESTORMER_RESTORATION:
        return (
            RESTORMER_RESTORATION,
            NAFNET_RESTORATION,
            OPENCV_RESTORATION,
        )
    if backend == NAFNET_RESTORATION:
        return (NAFNET_RESTORATION, OPENCV_RESTORATION)
    return (OPENCV_RESTORATION,)


def restore_document_image(
    image,
    backend=AUTO_RESTORATION,
    mode="ocr",
    use_deconvolution=False,
    restoration_task=None,
    docres_task="end2end",
    fallback_to_opencv=True,
):
    requested_backend = normalize_restoration_backend(backend)
    task = restoration_task or docres_task
    attempted_backends = []
    errors = []
    try:
        if requested_backend == OPENCV_RESTORATION:
            return restore_with_opencv(image, mode=mode, use_deconvolution=use_deconvolution)

        fallback_chain = get_restoration_fallback_chain(requested_backend)
        if not fallback_to_opencv:
            fallback_chain = fallback_chain[:1]

        for candidate in fallback_chain:
            attempted_backends.append(candidate)
            try:
                if candidate == OPENCV_RESTORATION:
                    result = restore_with_opencv(
                        image,
                        mode=mode,
                        use_deconvolution=use_deconvolution,
                    )
                elif candidate == DOCRES_RESTORATION:
                    result = restore_with_docres(image, task=task)
                elif candidate == RESTORMER_RESTORATION:
                    result = restore_with_restormer(image, task=task)
                elif candidate == NAFNET_RESTORATION:
                    result = restore_with_nafnet(image, task=task)
                else:
                    raise ValueError(f"Unsupported restoration backend: {candidate}")

                if candidate == requested_backend or (
                    requested_backend == AUTO_RESTORATION and len(attempted_backends) == 1
                ):
                    return RestorationResult(
                        image=result.image,
                        backend=result.backend,
                        message=result.message,
                        requested_backend=requested_backend,
                        attempted_backends=tuple(attempted_backends),
                    )

                aggregated_error = " | ".join(errors)
                return RestorationResult(
                    image=result.image,
                    backend=result.backend,
                    message=result.message,
                    error=aggregated_error,
                    used_fallback=True,
                    requested_backend=requested_backend,
                    attempted_backends=tuple(attempted_backends),
                )
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

        return RestorationResult(
            image=image,
            backend=requested_backend,
            error=" | ".join(errors),
            requested_backend=requested_backend,
            attempted_backends=tuple(attempted_backends),
        )
    except Exception as exc:
        if fallback_to_opencv and requested_backend != OPENCV_RESTORATION:
            fallback_result = restore_with_opencv(
                image,
                mode=mode,
                use_deconvolution=use_deconvolution,
            )
            attempted = list(attempted_backends)
            if OPENCV_RESTORATION not in attempted:
                attempted.append(OPENCV_RESTORATION)
            error_message = " | ".join(errors + [f"unexpected: {exc}"])
            return RestorationResult(
                image=fallback_result.image,
                backend=fallback_result.backend,
                message=fallback_result.message,
                error=error_message,
                used_fallback=True,
                requested_backend=requested_backend,
                attempted_backends=tuple(attempted),
            )

        return RestorationResult(
            image=image,
            backend=requested_backend,
            error=" | ".join(errors + [f"unexpected: {exc}"]),
            requested_backend=requested_backend,
            attempted_backends=tuple(attempted_backends),
        )
