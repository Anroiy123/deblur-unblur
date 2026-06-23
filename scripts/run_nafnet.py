import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAFNET_DIR = PROJECT_ROOT / "external" / "NAFNet"
DEFAULT_NAFNET_OPTIONS = DEFAULT_NAFNET_DIR / "options" / "test" / "GoPro" / "NAFNet-width64.yml"
DEFAULT_MAX_WIDTH = 640


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NAFNet single-image inference and copy its output."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output image path expected by the app.")
    parser.add_argument(
        "--task",
        default="deblurring",
        help="Task alias kept for API compatibility. NAFNet adapter currently targets deblurring.",
    )
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_NAFNET_DIR),
        help="Path to the cloned NAFNet repository.",
    )
    parser.add_argument(
        "--opt",
        default=str(DEFAULT_NAFNET_OPTIONS),
        help="Path to the NAFNet test options .yml file.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=int(os.environ.get("NAFNET_MAX_WIDTH", DEFAULT_MAX_WIDTH)),
        help="Downscale input width before inference for CPU stability; 0 disables resizing.",
    )
    parser.add_argument("--timeout", type=int, default=180, help="Inference timeout in seconds.")
    return parser.parse_args()


def prepare_resized_input(input_path, workspace_dir, max_width):
    source = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Unable to read NAFNet input image: {input_path}")

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
        raise RuntimeError(f"Unable to write resized NAFNet input image: {resized_path}")
    return resized_path, (original_width, original_height)


def finalize_output(restored_output, output_path, original_size):
    restored = cv2.imread(str(restored_output), cv2.IMREAD_COLOR)
    if restored is None:
        raise RuntimeError(f"Unable to read NAFNet output image: {restored_output}")

    original_width, original_height = original_size
    if restored.shape[1] != original_width or restored.shape[0] != original_height:
        restored = cv2.resize(
            restored,
            (original_width, original_height),
            interpolation=cv2.INTER_CUBIC,
        )

    if not cv2.imwrite(str(output_path), restored):
        raise RuntimeError(f"Unable to write NAFNet output image: {output_path}")


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    repo_dir = Path(args.repo_dir).resolve()
    options_path = Path(args.opt).resolve()
    demo_script = repo_dir / "basicsr" / "demo.py"

    if not input_path.exists():
        raise FileNotFoundError(f"NAFNet input does not exist: {input_path}")
    if not demo_script.exists():
        raise FileNotFoundError(f"NAFNet basicsr/demo.py not found in: {repo_dir}")
    if not options_path.exists():
        raise FileNotFoundError(f"NAFNet options file not found: {options_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nafnet_out_") as temp_dir:
        prepared_input, original_size = prepare_resized_input(
            input_path,
            temp_dir,
            args.max_width,
        )
        temp_output_path = Path(temp_dir) / output_path.name
        command = [
            sys.executable,
            "-m",
            "basicsr.demo",
            "-opt",
            str(options_path),
            "--input_path",
            str(prepared_input),
            "--output_path",
            str(temp_output_path),
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
            raise RuntimeError(f"NAFNet failed with exit code {completed.returncode}: {message}")
        if not temp_output_path.exists():
            raise RuntimeError(f"NAFNet completed but did not create the output image: {temp_output_path}")

        finalize_output(temp_output_path, output_path, original_size)


if __name__ == "__main__":
    main()
