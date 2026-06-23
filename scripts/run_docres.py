import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCRES_DIR = PROJECT_ROOT / "external" / "DocRes"


def parse_args():
    parser = argparse.ArgumentParser(description="Run DocRes and copy its output to a requested path.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output image path expected by the app.")
    parser.add_argument("--task", default="end2end", help="DocRes task: dewarping, deshadowing, appearance, deblurring, binarization, end2end.")
    parser.add_argument("--docres-dir", default=str(DEFAULT_DOCRES_DIR), help="Path to the cloned DocRes repository.")
    parser.add_argument("--timeout", type=int, default=180, help="Inference timeout in seconds.")
    return parser.parse_args()


def expected_output_path(out_dir, input_path, task):
    source = Path(input_path)
    return Path(out_dir) / f"{source.stem}_{task}{source.suffix}"


def build_docres_environment(shim_dir):
    """Provide compatibility aliases without modifying the DocRes submodule."""
    shim_dir = Path(shim_dir)
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(
        "import numpy as np\n"
        "if 'bool' not in np.__dict__:\n"
        "    np.bool = bool\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    pythonpath_entries = [str(shim_dir)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return environment


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    docres_dir = Path(args.docres_dir).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"DocRes input does not exist: {input_path}")
    if not (docres_dir / "inference.py").exists():
        raise FileNotFoundError(f"DocRes inference.py not found in: {docres_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="docres_out_") as temp_dir:
        environment = build_docres_environment(Path(temp_dir) / "compat")
        command = [
            sys.executable,
            "inference.py",
            "--im_path",
            str(input_path),
            "--task",
            args.task,
            "--out_folder",
            temp_dir,
            "--save_dtsprompt",
            "0",
        ]
        completed = subprocess.run(
            command,
            cwd=str(docres_dir),
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"DocRes failed with exit code {completed.returncode}: {message}")

        docres_output = expected_output_path(temp_dir, input_path, args.task)
        if not docres_output.exists():
            candidates = sorted(
                Path(temp_dir).glob(f"*_{args.task}.*"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            candidates = [path for path in candidates if "_prompt" not in path.stem]
            if not candidates:
                raise RuntimeError(f"DocRes completed but no task output was found in: {temp_dir}")
            docres_output = candidates[0]

        shutil.copyfile(docres_output, output_path)


if __name__ == "__main__":
    main()
