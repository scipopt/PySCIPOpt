"""Copy runtime docstrings into the type stubs of a repaired wheel.

Usage: python add_wheel_stub_docstrings.py <directory with one wheel>
Skipped on Python <3.10, which docstring-adder requires.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

DOCSTRING_ADDER = "git+https://github.com/astral-sh/docstring-adder@8377e23ca0c1533bef3e744209709d6d783782b6"


def run(*args, **kwargs):
    subprocess.run([sys.executable, *args], check=True, **kwargs)


def main():
    if sys.version_info < (3, 10):
        print("Skipping stub docstrings: docstring-adder requires Python >=3.10")
        return

    wheel_dir = Path(sys.argv[1]).resolve()
    (wheel,) = wheel_dir.glob("*.whl")
    run("-m", "pip", "install", "wheel==0.48.0", "numpy", f"docstring-adder @ {DOCSTRING_ADDER}")

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run("-m", "wheel", "unpack", str(wheel), "-d", str(tmp / "unpacked"))
        (root,) = (tmp / "unpacked").iterdir()

        env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
        run("-c", "import pyscipopt.scip", env=env, cwd=tmp)
        run("-m", "add_docstrings", "--packages", str(root / "pyscipopt"), env=env, cwd=tmp)

        (tmp / "packed").mkdir()
        run("-m", "wheel", "pack", str(root), "-d", str(tmp / "packed"))
        (packed,) = (tmp / "packed").glob("*.whl")
        wheel.unlink()
        shutil.move(packed, wheel_dir / packed.name)


if __name__ == "__main__":
    main()
