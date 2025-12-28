#!/bin/bash -e

# Test the stubs for pyscipopt using stubtest
# This checks that the type hints in the stubs are consistent with the actual implementation
# Prerequisite: install mypy (which provides stubtest) in the same environment as pyscipopt
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -m mypy.stubtest \
  --allowlist "$SCRIPT_DIR/allowlist" \
  --allowlist "$SCRIPT_DIR/todo" \
  pyscipopt
