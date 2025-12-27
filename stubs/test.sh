#!/bin/bash -e

# Test the stubs for pyscipopt using stubtest
# This checks that the type hints in the stubs are consistent with the actual implementation
# Prerequisite: install mypy in same environment as pyscipopt and put stubtest in PATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
stubtest \
  --allowlist "$SCRIPT_DIR/.stubtest-allowlist" \
  --allowlist "$SCRIPT_DIR/.stubtest-allowlist-todo" \
  pyscipopt
