"""Turn this source tree into a renamed variant of PySCIPOpt, e.g. pyscipopt_debug.

Usage: python scripts/rename_package.py <variant>
where <variant> is e.g. "debug" or "exact". Produces the import name
pyscipopt_<variant> and the distribution name PySCIPOpt-<variant>.

Used by CI to build the variant wheels from an unmodified checkout. It rewrites
the tree in place, so never run it on a checkout you want to keep.
"""

import shutil
import sys
from pathlib import Path

variant = sys.argv[1]
new = f"pyscipopt_{variant}"
root = Path(__file__).resolve().parent.parent

pkg = root / "src" / "pyscipopt"
renamed = root / "src" / new
pkg.rename(renamed)

# generated C would still carry the old module name; force cythonize to regenerate it
(renamed / "scip.c").unlink(missing_ok=True)
shutil.rmtree(root / "build", ignore_errors=True)

# every internal reference is an absolute "pyscipopt..." import, so a plain rewrite suffices
sources = [p for p in renamed.rglob("*") if p.suffix in {".py", ".pyx", ".pxd", ".pxi", ".pyi"}]
sources += [p for p in (root / "tests").rglob("*.py")]
sources += [root / "setup.py", root / "pyproject.toml", root / "scripts" / "add_wheel_stub_docstrings.py"]
for path in sources:
    text = path.read_text()
    if "pyscipopt" in text:
        path.write_text(text.replace("pyscipopt", new))

# distribution name is CamelCase, so the rewrite above does not touch it
for path, old in [(root / "pyproject.toml", 'name = "PySCIPOpt"'), (root / "setup.py", 'name="PySCIPOpt"')]:
    text = path.read_text()
    assert old in text, f"{old!r} not found in {path.name}"
    path.write_text(text.replace(old, old.replace("PySCIPOpt", f"PySCIPOpt-{variant}")))

print(f"Renamed package to {new} (distribution PySCIPOpt-{variant})")
