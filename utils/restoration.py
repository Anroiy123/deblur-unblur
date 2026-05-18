from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2

from utils.enhancement import enhance_image

OPENCV_RESTORATION = "opencv"
DOCRES_RESTORATION = "docres"
SUPPORTED_RESTORATION_BACKENDS = (OPENCV_RESTORATION, DOCRES_RESTORATION)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCRES_SCRIPT = PROJECT_ROOT / "scripts" / "run_docres.py"
DEFAULT_DOCRES_WEIGHTS = (
    PROJECT_ROOT / "external" / "DocRes" / "checkpoints" / "docres.pkl",
    PROJECT_ROOT / "external" / "DocRes" / "data" / "MBD" / "checkpoint" / "mbd.pkl",
)


@dataclass(frozen=True)
class RestorationResult:
    image: object
    backend: str
    message: str = ""
    error: str = ""
    used_fallback: bool = False

    @property
    def ok(self):
        return not self.error


def normalize_restoration_backend(backend):
    if not backend:
        return OPENCV_RESTORATION

    normalized = str(backend).strip().lower()
    if normalized in ("opencv", "classical", "baseline"):
        return OPENCV_RESTORATION
    if normalized in ("docres", "ai", "document_ai"):
        return DOCRES_RESTORATION

    raise ValueError(f"Unsupported restoration backend: {backend}")


def is_docres_configured():
    """
    DocRes is an external repo/weights pipeline, so this app uses a command adapter.

    Configure it with:
        DOCRES_COMMAND="python path/to/inference.py --task {task} --input {input} --output {output}"
    """
    return bool(get_docres_command_template())

def get_docres_command_template():
    """
    Return the explicit DocRes command or the local project wrapper if installed.
    """
    explicit_command = os.environ.get("DOCRES_COMMAND")
    if explicit_command:
        return explicit_command

    if DEFAULT_DOCRES_SCRIPT.exists() and all(weight.exists() for weight in DEFAULT_DOCRES_WEIGHTS):
        return (
            f'"{sys.executable}" "{DEFAULT_DOCRES_SCRIPT}" '
            '--task {task} --input "{input}" --output "{output}"'
        )

    return ""


def restore_with_opencv(image, mode="ocr", use_deconvolution=False):
    restored = enhance_image(image, use_deconvolution=use_deconvolution, mode=mode)
    return RestorationResult(
        image=restored,
        backend=OPENCV_RESTORATION,
        message="OpenCV baseline restoration completed",
    )


def restore_with_docres(image, task="end2end", timeout_seconds=180):
    """
    Run a configured DocRes command adapter.

    The command must write a restored image to the {output} path.
    """
    command_template = get_docres_command_template()
    if not command_template:
        raise RuntimeError("DOCRES_COMMAND is not configured and local DocRes wrapper/weights were not found")

    with tempfile.TemporaryDirectory(prefix="deblur_docres_") as temp_dir:
        input_path = os.path.join(temp_dir, "input.png")
        output_path = os.path.join(temp_dir, "output.png")

        if not cv2.imwrite(input_path, image):
            raise RuntimeError("Unable to write temporary DocRes input image")

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
            raise RuntimeError(f"DocRes command failed with exit code {completed.returncode}: {stderr}")

        restored = cv2.imread(output_path, cv2.IMREAD_COLOR)
        if restored is None:
            raise RuntimeError("DocRes command completed but did not produce a readable output image")

        return RestorationResult(
            image=restored,
            backend=DOCRES_RESTORATION,
            message=f"DocRes restoration completed with task={task}",
        )


def restore_document_image(
    image,
    backend=OPENCV_RESTORATION,
    mode="ocr",
    use_deconvolution=False,
    docres_task="end2end",
    fallback_to_opencv=True,
):
    """
    Restore a document/card image with either local OpenCV baseline or DocRes adapter.
    """
    backend = normalize_restoration_backend(backend)

    if backend == OPENCV_RESTORATION:
        return restore_with_opencv(image, mode=mode, use_deconvolution=use_deconvolution)

    try:
        return restore_with_docres(image, task=docres_task)
    except Exception as exc:
        if not fallback_to_opencv:
            return RestorationResult(image=image, backend=backend, error=str(exc))

        fallback = restore_with_opencv(image, mode=mode, use_deconvolution=use_deconvolution)
        return RestorationResult(
            image=fallback.image,
            backend=fallback.backend,
            message=fallback.message,
            error=str(exc),
            used_fallback=True,
        )
