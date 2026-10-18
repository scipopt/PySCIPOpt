"""Copy runtime docstrings into the type stubs bundled in a release wheel.

Run after wheel repair has bundled the native libraries. Install the tools and
NumPy (needed to import PySCIPOpt), unpack the wheel, copy docstrings from its
compiled extension into scip.pyi, and repack it with updated file hashes.

Skip Python <3.10 because docstring-adder requires Python >=3.10.
"""

import argparse
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "wheel_dir",
        type=Path,
        help="Directory containing exactly one wheel with its native libraries bundled",
    )
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        print("Skipping wheel stub docstrings: docstring-adder requires Python >=3.10")
        return

    wheel_dir = args.wheel_dir.resolve()
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected exactly one wheel in {wheel_dir}, found {len(wheels)}"
        )
    (wheel_path,) = wheels

    requirements_path = (
        Path(__file__).resolve().parents[1] / "requirements/wheel-stub-docstrings.txt"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ],
        check=True,
    )

    with TemporaryDirectory() as temporary_dir:
        unpack_dir = Path(temporary_dir)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "unpack",
                str(wheel_path),
                "-d",
                str(unpack_dir),
            ],
            check=True,
        )
        (wheel_root,) = unpack_dir.iterdir()
        # Put the unpacked wheel first on Python's import path so extraction uses
        # its compiled extension and bundled libraries.
        extractor_env = dict(
            os.environ,
            PYTHONPATH=str(wheel_root),
            PYTHONDONTWRITEBYTECODE="1",
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "add_docstrings",
                "--packages",
                str(wheel_root / "pyscipopt"),
            ],
            cwd=temporary_dir,
            env=extractor_env,
            check=True,
        )
        # The extractor reports success even when it cannot import the extension.
        # Require a known method's docstring before replacing the original wheel.
        stub_ast = ast.parse(
            (wheel_root / "pyscipopt/scip.pyi").read_text(encoding="utf-8")
        )
        model_class = next(
            (
                node
                for node in stub_ast.body
                if isinstance(node, ast.ClassDef) and node.name == "Model"
            ),
            None,
        )
        add_var_method = next(
            (
                node
                for node in getattr(model_class, "body", [])
                if isinstance(node, ast.FunctionDef) and node.name == "addVar"
            ),
            None,
        )
        if add_var_method is None or not ast.get_docstring(add_var_method):
            raise RuntimeError(
                f"Stub docstring extraction failed for {wheel_path.name}: "
                "Model.addVar has no docstring; see extractor output above"
            )

        # Repacking updates RECORD, the wheel's list of files and their hashes.
        pack_dir = unpack_dir / "packed"
        pack_dir.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                str(wheel_root),
                "-d",
                str(pack_dir),
            ],
            check=True,
        )
        (packed_wheel,) = pack_dir.glob("*.whl")
        wheel_path.unlink()
        shutil.move(packed_wheel, wheel_dir / packed_wheel.name)


if __name__ == "__main__":
    main()
