import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESTORMER_DIR = PROJECT_ROOT / "external" / "Restormer"
DEFAULT_MAX_WIDTH = 640
RESTORMER_TASK_ALIASES = {
    "deblurring": "Motion_Deblurring",
    "motion_deblurring": "Motion_Deblurring",
    "motion": "Motion_Deblurring",
    "defocus_deblurring": "Single_Image_Defocus_Deblurring",
    "defocus": "Single_Image_Defocus_Deblurring",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Restormer single-image inference and copy its output."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output image path expected by the app.")
    parser.add_argument(
        "--task",
        default="deblurring",
        help="App-level task alias. Defaults to deblurring -> Motion_Deblurring.",
    )
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_RESTORMER_DIR),
        help="Path to the cloned Restormer repository.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=int(os.environ.get("RESTORMER_MAX_WIDTH", DEFAULT_MAX_WIDTH)),
        help="Downscale input width before inference for CPU stability; 0 disables resizing.",
    )
    parser.add_argument("--timeout", type=int, default=180, help="Inference timeout in seconds.")
    return parser.parse_args()


def resolve_task_name(task):
    normalized = str(task or "").strip().lower()
    if normalized in RESTORMER_TASK_ALIASES:
        return RESTORMER_TASK_ALIASES[normalized]
    return task


def select_output_file(result_dir, input_path):
    input_path = input_path.resolve()
    direct_match = result_dir / input_path.name
    if direct_match.exists() and direct_match.resolve() != input_path:
        return direct_match

    candidates = sorted(
        [
            path
            for path in result_dir.rglob("*")
            if path.is_file() and path.resolve() != input_path
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    image_candidates = [
        path for path in candidates if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ]
    if image_candidates:
        return image_candidates[0]
    raise RuntimeError(f"Restormer completed but no output image was found in: {result_dir}")


def prepare_resized_input(input_path, workspace_dir, max_width):
    source = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Unable to read Restormer input image: {input_path}")

    original_height, original_width = source.shape[:2]
    if max_width <= 0 or original_width <= max_width:
        return input_path, (original_width, original_height)

    scale = max_width / original_width
    resized_height = max(1, int(original_height * scale))
    resized = cv2.resize(
        source,
        (max_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    resized_path = Path(workspace_dir) / input_path.name
    if not cv2.imwrite(str(resized_path), resized):
        raise RuntimeError(f"Unable to write resized Restormer input image: {resized_path}")
    return resized_path, (original_width, original_height)


def finalize_output(restored_output, output_path, original_size):
    restored = cv2.imread(str(restored_output), cv2.IMREAD_COLOR)
    if restored is None:
        raise RuntimeError(f"Unable to read Restormer output image: {restored_output}")

    original_width, original_height = original_size
    if restored.shape[1] != original_width or restored.shape[0] != original_height:
        restored = cv2.resize(
            restored,
            (original_width, original_height),
            interpolation=cv2.INTER_CUBIC,
        )

    if not cv2.imwrite(str(output_path), restored):
        raise RuntimeError(f"Unable to write Restormer output image: {output_path}")


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    repo_dir = Path(args.repo_dir).resolve()
    demo_script = repo_dir / "demo.py"

    if not input_path.exists():
        raise FileNotFoundError(f"Restormer input does not exist: {input_path}")
    if not demo_script.exists():
        raise FileNotFoundError(f"Restormer demo.py not found in: {repo_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_name = resolve_task_name(args.task)

    with tempfile.TemporaryDirectory(prefix="restormer_out_") as temp_dir:
        prepared_input, original_size = prepare_resized_input(
            input_path,
            temp_dir,
            args.max_width,
        )
        command = [
            sys.executable,
            "demo.py",
            "--task",
            task_name,
            "--input_dir",
            str(prepared_input),
            "--result_dir",
            temp_dir,
        ]
        completed = subprocess.run(
            command,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"Restormer failed with exit code {completed.returncode}: {message}")

        restored_output = select_output_file(Path(temp_dir), prepared_input)
        finalize_output(restored_output, output_path, original_size)


if __name__ == "__main__":
    main()
